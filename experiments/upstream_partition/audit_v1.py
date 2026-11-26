"""Independently audit all roles, input checkpoints, budget decisions, counts and timing pairs."""
import json
import itertools
import platform
import sys
import hashlib
from inputs import ROOT, np, pd, sha, write_json, tf
from evaluate import read, FAMILIES
from verify_results import check_policy, mask_for
from cohorts import ROLES
from paired_time import conditions
from provenance import verify_seal
from sklearn.metrics import average_precision_score

def main():
    a=pd.read_pickle(ROOT/"assignment.pkl")
    assert a.index.is_unique and a.groupby("domain").domain_role.nunique().max()==1
    report={}
    for split in ("random","domain"):
        batch=read(ROOT/("batch-"+split+".json"))["ids"]
        assert len(batch)==len(set(batch))==1000
        timings=pd.read_csv(ROOT/("timing-"+split+".csv"))
        for rep,g in timings.groupby("repeat"):
            assert sorted(g.order)==list(range(1,len(g)+1))
            assert not g.duplicated(["kind","family","policy","tau"]).any()
        residual=timings.total_ms-timings[["score_ms","route_ms","input_ms","gru_ms"]].sum(axis=1)
        assert np.abs(residual).max()<1e-8
        for k in conditions(split):
            out=ROOT/k;verify_seal(k)
            parts={r:pd.read_pickle(out/(r+".pkl")) for r in ROLES}
            for role,d in parts.items():
                assert (a.loc[d.url,split+"_role"]==role).all()
                assert np.array_equal(a.loc[d.url,"domain"],d.domain)
                assert d.index.is_unique and d.url.is_unique
                assert np.array_equal(1-d.type,a.loc[d.url,"y_mal"])
            for r1,r2 in itertools.combinations(ROLES,2):
                assert not set(parts[r1].url).intersection(parts[r2].url)
                if split=="domain": assert not set(parts[r1].domain).intersection(parts[r2].domain)
            test=parts["test"]
            assert set(batch)<=set(test.index)
            with np.load(out/"scores.npz") as s:
                ids,y,gru=s["id"],s["y"],s["gru"]
                assert np.array_equal(ids,test.index) and np.array_equal(y,1-test.type)
                results=read(out/"results.json")["policies"]
                check_policy(y,gru,np.ones(len(y),bool),results["gru"])
                for fam in FAMILIES:
                    p=s[fam]
                    ap=read(out/"results.json")["first_stage"][fam]["pr_auc_ap"]
                    assert abs(ap-average_precision_score(y,p))<1e-10
                    tau=read(out/"routing.json")["families"][fam]["tau"]
                    early=p<tau
                    for policy in ("basic","random","boundary"):
                        r=results[fam+"_"+policy]
                        mask=mask_for(ids,p,tau,0 if policy=="basic" else .05,policy)
                        if isinstance(mask,tuple):mask=mask[0]
                        assert int((mask & early).sum())==r["rechecks"]
                        check_policy(y,gru,mask,r)
            tr=read(out/"trained.json")
            assert len(tr["history"]["loss"])==20
            assert tr["weights_sha256"]==sha(out/"gru.h5") and tr["params"]==162481
            report[k]=dict(roles_verified=list(ROLES),test_rows=len(test),policies=7,weights=tr["weights_sha256"],validation_selected_epoch=tr["epoch"])
    for k in ("author-validation","author-final"):
        verify_seal(k)
        rec=read(ROOT/k/"trained.json")
        assert len(rec["history"]["loss"])==20 and rec["weights_sha256"]==sha(ROOT/k/"gru.h5")
    assert len(pd.read_csv(ROOT/"results/main-28.csv"))==28
    write_json(ROOT/"results/verification.json",dict(status="PASS",conditions=report,python=sys.version,tensorflow=tf.__version__,numpy=np.__version__,system=platform.platform(),timing_repeats=30,batch=1000,source_assignment=sha(ROOT/"assignment.csv.gz")))
if __name__=="__main__": main()
