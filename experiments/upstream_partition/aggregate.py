"""Recompute held-out counts and confidence intervals from row-level scores and paired timings."""
import json
import itertools
from pathlib import Path
from math import sqrt
from statistics import NormalDist
from collections import Counter
from inputs import ROOT, np, pd, sha, write_json
from evaluate import route, read, configurations, FAMILIES, TAUS
from experiment import cascade_metrics
from paired_time import conditions
from cohorts import ROLES


def loss_interval(y,additional,domain,split,always_same):
    if always_same:
        return 0.,0.
    hit=y.astype(bool)
    flags=additional[hit]
    n=len(flags); p=flags.mean()
    if split=="random":
        z=NormalDist().inv_cdf(.975)
        center=(p+z*z/(2*n))/(1+z*z/n)
        radius=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return max(0.,center-radius), min(1.,center+radius)
    # An empirical all-zero cluster distribution cannot estimate an upper bound.
    if not flags.any():
        return 0.,np.nan
    # Sample registered domains with replacement, keeping all malicious URLs in each.
    ix,_=pd.factorize(domain[hit],sort=True)
    pairs=np.column_stack([np.bincount(ix,weights=flags).astype(int), np.bincount(ix)])
    uniq,freq=np.unique(pairs,axis=0,return_counts=True)
    rng=np.random.default_rng(11024)
    sample=rng.multinomial(len(pairs),freq/freq.sum(),size=4000)
    rates=(sample@uniq[:,0])/(sample@uniq[:,1])
    return tuple(np.quantile(rates,[.025,.975]))


def load_test(k):
    out=ROOT/k
    d=pd.read_pickle(out/"test.pkl")
    with np.load(out/"scores.npz") as arr:
        s={a:arr[a] for a in arr.files}
    assert np.array_equal(d.index,s["id"]) and np.array_equal(1-d.type,s["y"])
    return d,s,{k:v["tau"] for k,v in read(out/"routing.json")["families"].items()}


def summarize_timing(split):
    d=pd.read_csv(ROOT/f"timing-{split}.csv")
    assert set(d.repeat)==set(range(1,31))
    rng=np.random.default_rng(509)
    draws=rng.integers(0,30,(10000,30))
    rows=[]
    for (kind,fam,pol,tau),g in d.groupby(["kind","family","policy","tau"]):
        g=g.sort_values("repeat")
        ref=d[(d.kind==kind)&(d.family=="gru")].sort_values("repeat")
        assert np.array_equal(g.repeat,ref.repeat) and g.calls.nunique()==1
        diff=ref.total_ms.to_numpy()-g.total_ms.to_numpy()
        med=np.median(diff)
        lo,hi=np.quantile(np.median(diff[draws],axis=1),[.025,.975])
        rows.append(dict(kind=kind,family=fam,policy=pol,tau=tau,main=g.main.all(),name="gru" if fam=="gru" else fam+"_"+pol,
                         calls=int(g.calls.iloc[0]),saved_ms=med,saved_lo=lo,saved_hi=hi,saved_pct=np.median(100*diff/ref.total_ms),
                         min_ms=g.total_ms.min(),max_ms=g.total_ms.max(),
                         **{k:g[k].median() for k in ("total_ms","input_ms","score_ms","route_ms","gru_ms")}))
    return pd.DataFrame(rows)


def cross_condition_times(split):
    raw=pd.read_csv(ROOT/f"timing-{split}.csv")
    active=raw.loc[raw.main].set_index(["kind","family","policy","repeat"]).sort_index()
    ref,ref_strip,team,team_strip=conditions(split)
    comparisons=[("remove_prefix",ref,ref_strip),("remove_prefix",team,team_strip),("change_cleaning",ref,team)]
    draws=np.random.default_rng(509).integers(0,30,(10000,30))
    result=[]
    for question,base,changed in comparisons:
        before,after=active.loc[base],active.loc[changed]
        assert before.index.equals(after.index)
        for (family,policy),group in (after.total_ms-before.total_ms).groupby(level=["family","policy"]):
            assert np.array_equal(group.index.get_level_values("repeat"),np.arange(1,31))
            values=group.to_numpy()
            lo,hi=np.quantile(np.median(values[draws],axis=1),[.025,.975])
            result.append(dict(split=split,comparison=question,base=base,changed=changed,
                               name="gru" if family=="gru" else family+"_"+policy,
                               delta_ms=np.median(values),lo_ms=lo,hi_ms=hi,
                               direction="유보" if lo<=0<=hi else "단축" if hi<0 else "증가"))
    return result


def score_scope(k,d,s,taus,ids,scope,grid=False):
    split="domain" if "domain" in k else "random"
    i=d.index.get_indexer(ids)
    assert min(i)>=0
    y=s["y"][i]; g=s["gru"][i]; names=s["id"][i]; domains=d.domain.to_numpy()[i]
    first={f:s[f][i] for f in FAMILIES}
    masks,rows=configurations(y,names,g,first,taus)
    expanded=[("gru" if n=="gru" else n.split("_")[0],"basic" if n=="gru" else n.split("_")[1],0 if n=="gru" else taus[n.split("_")[0]],masks[n],r,True) for n,r in rows.items()]
    if grid:
        for fam in FAMILIES:
            for t in TAUS:
                if t==taus[fam]: continue
                m,_=route(first[fam],names,t,0,"basic")
                expanded.append((fam,"basic",t,m,cascade_metrics(y,g,m),False))
    for fam,pol,t,mask,r,main in expanded:
        lo,hi=loss_interval(y,(g>.5)&~mask,domains,split,fam=="gru" or t==0)
        assert r.get("tau",t)==t
        yield {**r,"kind":k,"scope":scope,"name":"gru" if fam=="gru" else fam+"_"+pol,
               "family":fam,"policy":pol,"tau":t,"main":main,"malicious":int(y.sum()),
               "benign":int(len(y)-y.sum()),"loss_lo":lo,"loss_hi":hi}


def main():
    dest=ROOT/"results"
    dest.mkdir(exist_ok=True)
    population=[]; scores=[]; times=[]; contrast=[]; prefix=[]; stages=[]; ablations=[]; curves=[]; seeds=[]; sensitivity=[]; roles=[]
    for split in ("random","domain"):
        ks=conditions(split)
        cache={k:load_test(k) for k in ks}
        a=ks[0]; b=ks[2]
        common=cache[a][0].index.intersection(cache[b][0].index).sort_values()
        c=(cache[a][0].loc[common].url==cache[b][0].loc[common].url)
        assert c.all() and cache[a][0].loc[common].url.is_unique
        common_urls=set(cache[a][0].loc[common].url)
        assert common_urls==set(cache[a][0].url).intersection(cache[b][0].url)
        protected=cache[a][0].loc[common,["url","type"]]
        pd.DataFrame({"id":common,"y_mal":1-protected.type.to_numpy()}).to_csv(dest/(split+"-common.csv.gz"),index=False)
        for k in ks:
            d,s,taus=cache[k]
            training=pd.read_pickle(ROOT/k/"train.pkl")
            validation=pd.read_pickle(ROOT/k/"validation.pkl")
            leaks=len(common_urls.intersection(training.url)) + len(common_urls.intersection(validation.url))
            dom_overlap=len(set(training.domain)&set(d.domain))
            assert leaks==0 and (split!="domain" or dom_overlap==0)
            for role,df in (("train",training),("validation",validation),("test",d)):
                roles.append(dict(kind=k,role=role,n=len(df),malicious=int((df.type==0).sum()),benign=int(df.type.sum()),domains=df.domain.nunique()))
            form_peers=[name for name in ks if name.endswith(k.split("-")[-1])]
            same_form=cache[form_peers[0]][0].index.intersection(cache[form_peers[1]][0].index)
            population.append(dict(kind=k,train=len(training),validation=len(validation),test=len(d),malicious=int(s["y"].sum()),benign=int((1-s["y"]).sum()),shared_main_test=len(common),shared_same_form_test=len(same_form),cross_leaks=leaks,common_train_test_domains=dom_overlap))
            del training,validation
            scores.extend(score_scope(k,d,s,taus,d.index,"full",k.endswith("original")))
            if k.endswith("original"):
                scores.extend(score_scope(k,d,s,taus,common,"common"))
            pair=ks[0 if k==ks[1] else 1] if k in ks[:2] else ks[2 if k==ks[3] else 3]
            pc=d.index.intersection(cache[pair][0].index).sort_values()
            assert set(d.loc[pc,"url"])==set(d.url).intersection(cache[pair][0].url)
            scores.extend(score_scope(k,d,s,taus,pc,"prefix_common"))
            controls=read(ROOT/k/"controls-final.json")
            for fam,r in read(ROOT/k/"results.json")["first_stage"].items():
                stages.append(dict(kind=k,family=fam,**r))
                if k.endswith("original"):
                    ablations.append(dict(kind=k,family=fam,**controls[fam+"_omit_columns"]))
                    curves.extend(dict(kind=k,family=fam,**rec) for rec in controls[fam+"_curve"])
            if k.endswith("original"):
                sup=read(ROOT/k/"sensitivity.json")
                for fam,r in sup.items():
                    seeds.extend(dict(kind=k,family=fam,**z) for z in r["random_seeds"])
                    sensitivity.extend(dict(kind=k,family=fam,**z) for z in r["sensitivity"])
                for flag,r in read(ROOT/k/"fixed-prefix.json")["pairs"].items():
                    for name,flips in r["flips"].items():
                        prefix.append(dict(kind=k,group=flag,name=name,**{key:r[key] for key in r if key!="flips"},**flips))
        times.append(summarize_timing(split))
    perf=pd.DataFrame(scores)
    timing=pd.concat(times,ignore_index=True)
    keys=["kind","family","policy","tau","main","name"]
    decisions=perf.query("scope=='full'").merge(timing,on=keys,how="inner",validate="one_to_one")
    def verdict(r):
        if r.family=="gru": return "기준"
        if pd.isna(r.loss_hi): return "유보"
        if r.saved_lo<=0<=r.saved_hi or r.loss_lo<=.01<=r.loss_hi: return "유보"
        if r.saved_lo>0 and r.loss_hi<.01: return "충족·단축"
        return "시간 또는 미탐 기준 미충족"
    decisions["decision"]=decisions.apply(verdict,axis=1)
    main=decisions[decisions.main & decisions.kind.str.endswith("original")]
    assert len(main)==28
    for k in main.kind.unique():
        official=read(ROOT/k/"results.json")["policies"]
        for r in main[main.kind==k].itertuples():
            for field in ("tp","fp","fn","tn","additional_misses","t2_calls"):
                assert official[r.name][field]==getattr(r,field)
    paired_times=[row for split in ("random","domain") for row in cross_condition_times(split)]
    for filename,table in (("main-28",main),("full-and-paired",perf),("timing-summary",timing),("timing-pairs",paired_times),("threshold-decisions",decisions),("cohorts",population),("roles",roles),("prefix-fixed",prefix),("stage1",stages),("ablation",ablations),("learning",curves),("seeds",seeds),("sensitivity",sensitivity)):
        pd.DataFrame(table).to_csv(dest/(filename+".csv"),index=False)
    print("Verified main rows",len(main),flush=True)


if __name__=="__main__":
    main()
