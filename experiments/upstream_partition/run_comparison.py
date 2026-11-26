"""Run fresh cohorts, train checkpoints and validation-only stages, then freeze and score."""
import argparse
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ["URLCASCADE_RUN_DIR"]).resolve()
KINDS=[s+"-"+f for s in ("random","domain") for f in ("original","stripped")]
KINDS+=["team-"+k for k in KINDS]


def invoke(script,*args):
    log=ROOT/("-".join((script,*args)).replace("--","")+".log")
    with log.open("a",encoding="utf8") as handle:
        subprocess.run([sys.executable,"-u",str(HERE/(script+".py")),*args],stdout=handle,stderr=subprocess.STDOUT,check=True)


def train_fit(kind):
    if not (ROOT/kind/"trained.json").exists():
        invoke("train_gru",kind)
    if not (ROOT/kind/"stage1/t1-selected.json").exists():
        invoke("fit_stages",kind)
    if not (ROOT/kind/"routing.json").exists():
        invoke("evaluate",kind,"--freeze")


def main():
    futures=[]
    with ThreadPoolExecutor(max_workers=3) as workers:
        for form in ("original","stripped"):
            for proc in ("reference","team"):
                pair=[("team-" if proc=="team" else "")+s+"-"+form for s in ("random","domain")]
                if not all((ROOT/k/"cleaning.json").exists() for k in pair):
                    invoke("cohorts",proc,form)
                for k in pair:
                    futures.append(workers.submit(train_fit,k))
        for work in futures:
            work.result()
    for kind in KINDS:
        if not (ROOT/kind/"evaluation-complete").exists():
            invoke("score_v1",kind)
    invoke("provenance","example")


if __name__=="__main__":
    main()
