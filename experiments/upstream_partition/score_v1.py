"""Frozen held-out scoring and paired perturbation on all original test rows."""
import sys
import json
from inputs import ROOT, PREFIX, np, pd, write_json, tf, encode, vectorizer, preprocess_link, extract_features, sha
from evaluate import (author, score, load_gru, load_stages, load_rows, configurations, supplemental, controls, metrics, read, FAMILIES)
from cohorts import model_url
from run_comparison import KINDS


def score_test(kind):
    for k in KINDS:
        directory=ROOT/k
        frozen=read(directory/"routing.json")
        assert frozen["validation_rows"]>0
        assert frozen["gru_sha256"]==sha(directory/"gru.h5")
        assert all(sha(directory/"stage1"/(f+".joblib"))==h for f,h in frozen["first_stage_sha256"].items())
    out=ROOT/kind
    stages=load_stages(out)
    model=load_gru(out)
    thresholds={k:v["tau"] for k,v in read(out/"routing.json")["families"].items()}
    d,y,z,x=load_rows(out,"test")
    ids=np.asarray(d.index,dtype=str)
    g=score(model,z,x)
    p={k:m.predict_proba(x)[:,1] for k,m in stages.items()}
    masks,base=configurations(y,ids,g,p,thresholds)
    for family in FAMILIES:
        assert base[family+"_random"]["rechecks"]==base[family+"_boundary"]["rechecks"]==int((p[family]<thresholds[family]).sum()*.05)
    controls(out,d,y,x,p)
    write_json(out/"results.json",dict(policies=base,first_stage={k:metrics(y,v,v>=.5) for k,v in p.items()}))
    np.savez_compressed(out/"scores.npz",id=ids,y=y,gru=g,**p)
    if kind.endswith("original"):
        write_json(out/"sensitivity.json",{k:supplemental(y,ids,g,v,thresholds[k]) for k,v in p.items()})
        layer=vectorizer(read(out/"vocab.json"))
        alt={}
        for form in ("original","stripped"):
            urls=d.url.map(lambda u:model_url(u,form,kind.startswith("team-")))
            xx=np.asarray([extract_features(u) for u in urls],np.float32)
            zz=encode(urls.map(preprocess_link),layer)
            if form=="original":
                np.testing.assert_array_equal(xx,x)
                np.testing.assert_array_equal(zz,z)
            else:
                gp=score(model,zz,xx)
                pp={k:m.predict_proba(xx)[:,1] for k,m in stages.items()}
                am,ar=configurations(y,ids,gp,pp,thresholds)
                has=d.url.str.match(PREFIX).to_numpy()
                for label,where in (("with_prefix",has),("without_prefix",~has)):
                    alt[label]=dict(rows=int(where.sum()),malicious=int(y[where].sum()),token_changes=int((where & np.any(zz!=z,axis=1)).sum()),numeric_changes=int((where & np.any(xx!=x,axis=1)).sum()),flips={})
                    for n in am:
                        a,b=(g>.5)&masks[n],(gp>.5)&am[n]
                        alt[label]["flips"][n]=dict(changed=int(((a!=b)&where).sum()),malicious_to_benign=int((a&~b&where).sum()),benign_to_malicious=int((~a&b&where).sum()))
                np.savez_compressed(out/"paired-scores.npz",id=ids,y=y,gru=gp,**pp)
                write_json(out/"fixed-prefix.json",dict(stripped=ar,pairs=alt))
    (out/"evaluation-complete").touch(exist_ok=False)


if __name__=="__main__":
    if sys.argv[1]=="author-validation":
        author(ROOT/sys.argv[1])
    else:
        score_test(sys.argv[1])
