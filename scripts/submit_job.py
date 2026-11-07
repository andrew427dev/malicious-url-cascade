"""Official jobsubmit only, maintenance guard, fresh Job IDs and permission snapshots."""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from zoneinfo import ZoneInfo


def permission_snapshot():
    out={}
    for p in (Path.home(),Path.home()/'.ssh',Path.home()/'.ssh/authorized_keys'):
        try:
            s=p.stat()
            out[str(p)]=dict(mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid)
        except OSError as exc:
            out[str(p)]=dict(error=str(exc))
    return out


def maintenance_overlap(now, duration):
    zone=ZoneInfo('Asia/Seoul')
    start=datetime(2026,11,7,9,0,tzinfo=zone)
    end=datetime(2026,11,9,0,0,tzinfo=zone)
    return now < end and now+timedelta(seconds=duration) >= start


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare-b','smoke','pilot'],required=True)
    p.add_argument('--minutes',type=int,required=True)
    p.add_argument('--cpu',type=int,default=1)
    p.add_argument('--memory',default='8G')
    p.add_argument('script_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    workspace=(Path.home()/'workspace').resolve()
    if workspace not in root.parents or root.name != 'compsec-url-cascade':
        raise RuntimeError('Use the separate course project inside ~/workspace.')
    if args.minutes<=0 or args.minutes>1440 or args.cpu<1 or not re.fullmatch(r'[1-9][0-9]*[MG]',args.memory):
        raise ValueError('invalid_resource_request')
    now=datetime.now(ZoneInfo('Asia/Seoul'))
    if maintenance_overlap(now,args.minutes*60):
        raise RuntimeError('Known maintenance overlaps the requested job. Recheck policy after maintenance.')
    # Requests are proposals, not proof of post-maintenance account limits.
    duration=f'{args.minutes//1440}-{args.minutes%1440//60:02}:{args.minutes%60:02}:00'
    cmd=['jobsubmit','-c',str(args.cpu),'-m',args.memory,'-t',duration]
    if args.stage!='prepare-b':
        cmd+=['-p','4000ada','-G','1']
    rest=args.script_args[1:] if args.script_args[:1]==['--'] else args.script_args
    cmd+=['bash',str(root/'scripts/run_job.sh'),args.stage,*rest]
    log=dict(time_kst=now.isoformat(),command=cmd,permissions_before=permission_snapshot())
    result=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
    log.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
               permissions_after=permission_snapshot())
    match=re.search(r'Submitted batch job\s+(\d+)',result.stdout)
    if match:
        log['job_id']=match.group(1)
        log['school_logs']=str(Path.home()/'jobs'/match.group(1))
    folder=root/'artifacts/submissions'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/(now.strftime('%Y%m%dT%H%M%S%f')+'.json')).open('x') as f:
        json.dump(log,f,indent=2)
    print(json.dumps(log,indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)
    if not match:
        raise RuntimeError('Job ID not parsed. Inspect saved output; do not blindly resubmit.')


if __name__=='__main__':
    main()
