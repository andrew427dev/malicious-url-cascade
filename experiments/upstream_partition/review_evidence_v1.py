"""Capture retrospective provenance, exact input overlap and paired-analysis checks."""
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(os.environ["URLCASCADE_RUN_DIR"]).resolve()
HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def signatures(out,role):
    tok=np.load(out/(role+"-tokens.npy"),mmap_mode="r")
    num=np.load(out/(role+"-features.npy"),mmap_mode="r")
    return [hashlib.sha256(t.tobytes()+n.tobytes()).digest() for t,n in zip(tok,num)]

def main():
    out=ROOT/"results"
    roles=ROOT/"assignment.csv.gz"
    files={name:digest(HERE/name) for name in ("assign_roles.py","cohorts.py")}
    files[roles.name]=digest(roles)
    assert files[roles.name]==json.loads((ROOT/"source-manifest.json").read_text())["assignment_sha256"]
    av=ROOT/"author-validation"
    af=ROOT/"author-final"
    traces={}
    for p in (av/"training-inputs.json",av/"trained.json",av/"external-scores.npz",af/"training-inputs.json",af/"trained.json"):
        traces[p.relative_to(ROOT).as_posix()]=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).astimezone().isoformat()
    model_hashes={d.name:digest(d/"gru.h5") for d in (av,af)}
    assert len(set(model_hashes.values()))==2
    overlaps=[]
    for split in ("random","domain"):
        common=pd.read_csv(out/(split+"-common.csv.gz"))["id"]
        for prefix in ("","team-"):
            kind=prefix+split+"-original";folder=ROOT/kind
            prior=set(signatures(folder,"train"))|set(signatures(folder,"validation"))
            hits=np.asarray([s in prior for s in signatures(folder,"test")])
            ids=pd.read_csv(folder/"test-rows.csv.gz")["id"]
            overlaps.append(dict(kind=kind,full=int(hits.sum()),common=int(hits[ids.isin(common)].sum())))
    paired=pd.read_csv(out/"timing-pairs.csv")
    assert len(paired)==42
    policy_pairs=[]
    for split in ("random","domain"):
        raw=pd.read_csv(ROOT/f"timing-{split}.csv")
        raw=raw[raw.main].copy()
        raw["name"]=np.where(raw.family=="gru","gru",raw.family+"_"+raw.policy)
        for r in paired[paired.split==split].itertuples():
            part=raw[raw.name==r.name].pivot(index="repeat",columns="kind",values="total_ms")
            assert len(part)==30 and np.isclose((part[r.changed]-part[r.base]).median(),r.delta_ms)
        fam="boost" if split=="random" else "logistic"
        block=raw[(raw.kind==split+"-original") & (raw.family==fam)].pivot(index="repeat",columns="policy",values="total_ms")
        differences=(block.boundary-block.basic).to_numpy()
        draws=np.random.default_rng(509).integers(0,len(differences),(10000,len(differences)))
        lo,hi=np.quantile(np.median(differences[draws],axis=1),[.025,.975])
        policy_pairs.append(dict(split=split,family=fam,boundary_minus_basic_ms=np.median(differences),lo_ms=lo,hi_ms=hi))
    pd.DataFrame(policy_pairs).to_csv(out/"configuration-pairs.csv",index=False)
    grid=pd.read_csv(out/"threshold-decisions.csv")
    identical=(grid.family=="gru") | (grid.tau==0)
    assert grid[identical][["loss_lo","loss_hi"]].eq(0).all().all()
    unknown=grid.loss_hi.isna()
    assert grid[unknown].decision.eq("유보").all()
    record=dict(checked=datetime.now().astimezone().isoformat(),source_hash_timing="after_training",
                hashes=files,author_event_times=traces,author_weight_sha256=model_hashes,
                external_scores_model="author-validation",input_overlap=overlaps,paired_contrasts=len(paired),
                indeterminate_loss_upper_bounds=int(unknown.sum()))
    (out/"review-evidence.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(record,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
