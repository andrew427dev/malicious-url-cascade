"""Full, sequential entry point from a new destination, without historical artifacts."""
import sys
import os
import subprocess
from pathlib import Path

def main():
    if len(sys.argv)!=2: raise SystemExit("python rerun_v1.py NEW_DIRECTORY")
    out=Path(sys.argv[1]).resolve()
    if out.exists():raise FileExistsError(out)
    here=Path(__file__).resolve().parent
    os.environ["URLCASCADE_RUN_DIR"]=str(out)
    os.environ["PYTHONUTF8"]="1"
    def run(name,*args):
        subprocess.run([sys.executable,"-u",str(here/(name+".py")),*map(str,args)],check=True)
    run("fetch",out)
    run("inputs","parity")
    run("assign_roles")
    for k in ("author-validation","author-final"):
        run("inputs",k)
        run("train_gru",k)
        if k=="author-validation":run("score_v1",k)
    run("run_comparison")
    run("finish_v1")
if __name__=="__main__":main()
