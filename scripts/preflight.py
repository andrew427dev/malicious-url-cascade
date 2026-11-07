"""Read-only server inventory, safe on the login node; no training, no chmod."""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return dict(argv=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(argv=args, error=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    home = Path.home()
    permissions = {}
    for p in (home, home/'.ssh', home/'.ssh/authorized_keys'):
        try:
            s = p.stat()
            permissions[str(p)] = dict(mode=oct(stat.S_IMODE(s.st_mode)), uid=s.st_uid,
                                       gid=s.st_gid, group_writable=bool(s.st_mode & stat.S_IWGRP))
        except OSError as e:
            permissions[str(p)] = dict(error=str(e))
    report = dict(time_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  python=sys.version, platform=platform.platform(), permissions=permissions,
                  thesis_exists=(home/'workspace/korean-rag-thesis').is_dir(),
                  course_path_exists=(home/'workspace/compsec-url-cascade').exists(),
                  modules={m:importlib.util.find_spec(m) is not None for m in ('venv','ensurepip','pip')},
                  executables={m:shutil.which(m) for m in ('python3','git','conda','jobsubmit','jobexec','nvidia-smi','nvcc')},
                  commands=[command(a) for a in ([sys.executable,'-m','pip','--version'],['git','--version'],
                    ['quota','-s'],['squeue','-u',os.environ.get('USER','')],['jobsubmit','-h'],['ldconfig','-p'])])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
