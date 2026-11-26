"""Thirty blocked, randomized, paired raw-URL timings including a measured gate sweep."""
import argparse
from time import perf_counter_ns
from collections import defaultdict
from threadpoolctl import threadpool_limits, threadpool_info
from inputs import ROOT, tf, np, pd, vectorizer, preprocess_link, extract_features, write_json
from evaluate import load_gru, load_stages, load_rows, FAMILIES, POLICIES, read, route, TAUS
from cohorts import model_url


def conditions(split):
    return [p+split+"-"+v for p in ("","team-") for v in ("original","stripped")]


def shared_batch(split):
    common=None
    for kind in conditions(split):
        ids=pd.read_csv(ROOT/kind/"test-rows.csv.gz").id
        common=set(ids) if common is None else common.intersection(ids)
    ids=np.random.default_rng(46).choice(sorted(common),1000,replace=False)
    return ids,len(common)


def ready(kind,ids):
    out=ROOT/kind
    frame,y,z,x=load_rows(out,"test")
    loc=frame.index.get_indexer(ids)
    assert np.all(loc>=0)
    raw=frame.url.iloc[loc].tolist()
    x,z=x[loc],z[loc]
    with np.load(out/"scores.npz",allow_pickle=False) as saved:
        assert np.array_equal(ids,saved["id"][loc]) and np.array_equal(y[loc],saved["y"][loc])
        expected={k:saved[k][loc] for k in ("gru",*FAMILIES)}
    layer=vectorizer(read(out/"vocab.json"))
    model=load_gru(out)
    gates=load_stages(out)
    form=kind.split("-")[-1]
    team=kind.startswith("team-")
    taus={k:v["tau"] for k,v in read(out/"routing.json")["families"].items()}
    specs=[("gru","basic",0,0.)]
    for fam in FAMILIES:
        specs.extend((fam,pol,r,taus[fam]) for pol,r in POLICIES)
        if form=="original":
            specs.extend((fam,"basic",0,t) for t in TAUS if t!=taus[fam])
    expected_mask={s:np.ones(len(ids),bool) if s[0]=="gru" else route(expected[s[0]],ids,s[3],s[2],s[1])[0] for s in specs}
    expected_pred={s:(expected["gru"]>.5)&v for s,v in expected_mask.items()}

    @tf.function(input_signature=[tf.TensorSpec([None,50],tf.int32),tf.TensorSpec([None,18],tf.float32)])
    def forward(tok,num):
        return model([tok,num],training=False)

    def execute(spec):
        fam,policy,rate,tau=spec
        a=perf_counter_ns()
        urls=[model_url(u,form,team) for u in raw]
        numeric=np.asarray([extract_features(u) for u in urls],np.float32)
        b=perf_counter_ns()
        scores=None if fam=="gru" else gates[fam].predict_proba(numeric)[:,1]
        c=perf_counter_ns()
        mask=np.ones(len(raw),bool) if fam=="gru" else route(scores,ids,tau,rate,policy)[0]
        idx=np.flatnonzero(mask)
        pred=np.zeros(len(raw),bool)
        d=perf_counter_ns()
        tok=tf.cast(layer(tf.constant([preprocess_link(urls[i]) for i in idx])),tf.int32) if len(idx) else None
        e=perf_counter_ns()
        if len(idx):
            pred[idx]=forward(tok,numeric[idx]).numpy().ravel()<.5
        f=perf_counter_ns()
        np.testing.assert_array_equal(x,numeric)
        if tok is not None:
            np.testing.assert_array_equal(z[idx],tok.numpy())
        np.testing.assert_array_equal(expected_mask[spec],mask)
        np.testing.assert_array_equal(expected_pred[spec],pred)
        return dict(kind=kind,family=fam,policy=policy,tau=tau,rate=rate,main=fam=="gru" or tau==taus[fam],calls=len(idx),
                    input_ms=(b-a+e-d)/1e6,score_ms=(c-b)/1e6,route_ms=(d-c)/1e6,gru_ms=(f-e)/1e6,total_ms=(f-a)/1e6)
    return specs,execute


def main(split):
    with threadpool_limits(limits=1):
        ids,n=shared_batch(split)
        write_json(ROOT/("batch-"+split+".json"),dict(ids=ids.tolist(),eligible=n,threadpools=threadpool_info()))
        tasks=[]
        for kind in conditions(split):
            specs,execute=ready(kind,ids)
            for spec in specs:
                execute(spec)
                tasks.append((execute,spec))
        rng=np.random.default_rng(47)
        rows=[]
        for rep in range(1,31):
            for order,idx in enumerate(rng.permutation(len(tasks)),1):
                call,spec=tasks[idx]
                rows.append(dict(repeat=rep,order=order,**call(spec)))
            print(split,rep,flush=True)
        df=pd.DataFrame(rows)
        assert len(df)==30*len(tasks)
        df.to_csv(ROOT/("timing-"+split+".csv"),index=False)


if __name__=="__main__":
    import sys
    main(sys.argv[1])
