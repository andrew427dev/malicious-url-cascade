"""Run the author evaluation and serial timing only after all training/evaluation jobs finish."""
import os
import sys
from pathlib import Path
from time import sleep
from run_comparison import ROOT,KINDS,invoke
from concurrent.futures import ThreadPoolExecutor

def wait_for(path):
    while not path.exists():
        sleep(20)

def author_evaluation():
    wait_for(ROOT/"author-validation/trained.json")
    if not (ROOT/"author-validation/results.json").exists():
        invoke("score_v1","author-validation")

def main():
    with ThreadPoolExecutor(max_workers=1) as pool:
        author=pool.submit(author_evaluation)
        for k in KINDS: wait_for(ROOT/k/"evaluation-complete")
        wait_for(ROOT/"author-final/trained.json")
        author.result()
    wait_for(ROOT/"feature-example.json")
    for split in ("random","domain"):
        if not (ROOT/("timing-"+split+".csv")).exists():invoke("paired_time",split)
    invoke("aggregate")
    invoke("audit_v1")
    invoke("review_evidence_v1")
    invoke("demo_v1")
    invoke("demo_screen_v1")
    print("Finished performance, timing and independent checks",flush=True)

if __name__=="__main__":main()
