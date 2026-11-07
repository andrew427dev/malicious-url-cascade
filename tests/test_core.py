import itertools
import math
import unittest
from urlcascade.core import infer, binary_metrics, cascade_invariants, select_gate, hostname_for_grouping
from urlcascade.features import extract_features, FEATURE_NAMES
from urlcascade.adapters import preprocess_link
from urlcascade.data import audit, exclude_external, validate_partitions


def rows(n=4):
    return [dict(input_id=str(i),raw_url='https://example.com/a'+str(i)) for i in range(n)]


class CoreTests(unittest.TestCase):
    def test_feature_vector_and_order(self):
        self.assertEqual(len(FEATURE_NAMES),18)
        self.assertEqual(extract_features('https://www.example.com/a1?x=2'),
                         [0,1,2,1,0,1,0,1,1,0,1,0,1,30,15,2,20,2])

    def test_address_feature_matches_pinned_edge_cases(self):
        samples = {
            'http://example.org/0x7f.0x0.0x0.0x1/': 0,
            'http://example.org/0x7f.0x0.0x0.0x1/1:2:3:4:5:6:7:8': 1,
            'http://[2001:db8::1]/': 0,
            'http://example.org/255.255.255.255/': 1,
            'http://example.org/192.0.2.1?': 0,
        }
        for url, expected in samples.items():
            with self.subTest(url=url):
                self.assertEqual(extract_features(url)[0], expected)

    def test_original_feature_quirks_preserved(self):
        self.assertEqual(extract_features('http://192.0.2.1:80/a')[0],0)
        self.assertEqual(extract_features('https://example.com/192.0.2.1/')[0],1)
        self.assertEqual(extract_features('example.com/a')[14],0)
        self.assertEqual(extract_features('http://example.com:8080/a')[14],16)

    def test_preprocessor_exact(self):
        self.assertEqual(preprocess_link(' HTTPS://WWW.Example.com/A-1?q=Z '),'examplecoma1qz')

    def test_boundary_and_selective_call(self):
        seen=[]
        def heavy(urls,x):
            seen.append((urls,x))
            return [.5, math.nextafter(.5,0)]
        out=infer(rows(3),mode='cascade',t1=lambda x:[.009,.01,.9],t2=heavy,tau=.01)
        self.assertEqual([r['label_mal'] for r in out],[0,0,1])
        self.assertEqual([r['t2_called'] for r in out],[False,True,True])
        self.assertIsNone(out[0]['t2_score_mal'])
        self.assertEqual(seen[0][0],[r['raw_url'] for r in rows(3)[1:]])
        self.assertEqual(seen[0][1],[extract_features(r['raw_url']) for r in rows(3)[1:]])

    def test_tau_zero_equals_baseline(self):
        t2=lambda urls,x:[.499,.5,.8,0]
        baseline=infer(rows(),mode='t2',t2=t2)
        cascade=infer(rows(),mode='cascade',t1=lambda x:[0,.01,.5,1],t2=t2,tau=0)
        self.assertEqual([r['label_mal'] for r in baseline],[r['label_mal'] for r in cascade])
        self.assertTrue(all(r['t2_called'] for r in cascade))

    def test_all_early_empty_and_invalid_skip_models(self):
        def forbidden(*args):
            self.fail('unnecessary model call')
        self.assertEqual(infer([],mode='cascade',t1=forbidden,t2=forbidden),[])
        out=infer(rows(),mode='cascade',t1=lambda x:[0]*len(x),t2=forbidden,tau=.1)
        self.assertTrue(all(r['route']=='early_normal' for r in out))
        bad=[dict(input_id=str(i),raw_url=u) for i,u in enumerate([None,'','http:///','http://x:99999','a b'])]
        out=infer(bad,mode='cascade',t1=forbidden,t2=forbidden)
        self.assertTrue(all(r['status']=='invalid' and r['label_mal'] is None for r in out))

    def test_duplicate_urls_keep_order_in_batches(self):
        data=[dict(input_id=str(i),raw_url='https://example.com/a') for i in range(257)]
        for batch in (1,128):
            out=[]
            for start in range(0,len(data),batch):
                out.extend(infer(data[start:start+batch],mode='cascade',t1=lambda x:[.1]*len(x),
                                 t2=lambda u,x:[.2]*len(u),tau=.1))
            self.assertEqual([r['input_id'] for r in out],[r['input_id'] for r in data])
            self.assertTrue(all(r['label_mal']==1 for r in out))

    def test_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            infer([rows()[0],rows()[0]],mode='always_normal')

    def test_invalid_model_scores_are_errors(self):
        out=infer(rows(3),mode='t1',t1=lambda x:[math.nan,-1,1.1])
        self.assertTrue(all(r['status']=='error' and r['label_mal'] is None for r in out))
        out=infer(rows(1),mode='t2',t2=lambda u,x:[math.inf])
        self.assertEqual(out[0]['status'],'error')
        self.assertTrue(out[0]['t2_called'])

    def test_t1_exact_half_is_malicious(self):
        out=infer(rows(1),mode='t1',t1=lambda x:[.5])
        self.assertEqual(out[0]['label_mal'],1)

    def test_truth_table_subset_and_extra_false_negatives(self):
        table=list(itertools.product([0,1],repeat=3))
        y,b,called=map(list,zip(*table))
        c=[bp*route for bp,route in zip(b,called)]
        report=cascade_invariants(y,b,c,called)
        self.assertEqual(report['additional_fn'],1)
        self.assertEqual(report['early_malicious'],2)
        self.assertEqual(report['recall_drop'],.25)
        with self.assertRaises(AssertionError):
            cascade_invariants([1],[0],[1],[True])
        with self.assertRaises(AssertionError):
            cascade_invariants([1],[1],[0],[True])

    def test_gate_count_floor(self):
        self.assertFalse(cascade_invariants([1]*99,[1]*99,[0]+[1]*98,[False]+[True]*98)['allowed'])
        self.assertTrue(cascade_invariants([1]*100,[1]*100,[0]+[1]*99,[False]+[True]*99)['allowed'])

    def test_na_metrics_and_no_malicious_gate(self):
        self.assertIsNone(binary_metrics([0],[0])['recall'])
        self.assertIsNone(binary_metrics([1],[1])['fpr'])
        self.assertIsNone(binary_metrics([],[])['accuracy'])
        self.assertFalse(cascade_invariants([0],[0],[0],[False])['allowed'])

    def test_gate_timing_tie_prefers_smaller_tau(self):
        table=[dict(tau=.001,allowed=True,mean_latencies=[1.04]*5),
               dict(tau=.01,allowed=True,mean_latencies=[1.0]*5),
               dict(tau=.1,allowed=False,mean_latencies=[.1]*5)]
        self.assertEqual(select_gate(table),.001)

    def test_hostname_normalization(self):
        self.assertEqual(hostname_for_grouping('HTTPS://EXAMPLE.COM.:443/a'),'example.com')
        self.assertEqual(hostname_for_grouping('https://[2001:db8::1]:443/a'),'2001:db8::1')
        self.assertEqual(hostname_for_grouping('example.com:443/a'),'example.com')
        self.assertTrue(hostname_for_grouping('https://예제.kr').startswith('xn--'))
        for value in ('http://256.1.1.1/a','https://-bad.com','https://a..com','ftp://example.com'):
            with self.assertRaises(ValueError):
                hostname_for_grouping(value)

    def test_audit_conflicts_duplicates_and_raw_preservation(self):
        raw=[dict(input_id=str(i),raw_url=u,y_original=y,source='fixture') for i,(u,y) in enumerate([
            (' https://example.com/a ',1),('https://example.com/a',1),
            ('https://example.org/b',0),('https://example.org/b',1),
            ('https://example.net/c',9),('',0)])]
        clean,excluded,keys,groups=audit(raw,hostname_for_grouping)
        self.assertEqual(len(clean),1)
        self.assertEqual(clean[0]['raw_url'],' https://example.com/a ')
        self.assertEqual(clean[0]['url_key'],'https://example.com/a')
        self.assertEqual(clean[0]['source_rows'],['0','1'])
        self.assertEqual(sum(r['exclusion_reason']=='conflicting_label' for r in excluded),2)
        self.assertIn('example.org',groups)
        external=[dict(input_id='ext',raw_url='https://example.org/new',url_key='https://example.org/new',group_id='example.org')]
        kept,dropped=exclude_external(external,keys,groups)
        self.assertEqual(kept,[])
        self.assertEqual(dropped[0]['exclusion_reason'],'internal_domain_overlap')

    def test_partition_overlap_and_missing_classes_fail(self):
        data=[]
        for j,split in enumerate(('train','model_selection','gate_selection','test')):
            for y in (0,1):
                data.append(dict(input_id=f'{j}-{y}',split=split,y_mal=y,group_id=f'{j}-{y}.example'))
        self.assertEqual(validate_partitions(data)['train']['rows'],2)
        data[-1]['group_id']=data[0]['group_id']
        with self.assertRaisesRegex(ValueError,'domain_overlap'):
            validate_partitions(data)
        with self.assertRaisesRegex(ValueError,'missing_class'):
            validate_partitions(data[:1])


if __name__ == '__main__':
    unittest.main()
