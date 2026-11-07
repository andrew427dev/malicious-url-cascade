"""Portable cloud launcher. Run as a subprocess; do not mutate notebook kernel."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from urlcascade.runtime import compute_context

UV_VERSION = '0.9.30'  # Verified official PyPI release, bootstrap tool only.
MANAGED_PYTHON = '3.10.19'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def child_environment(work):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('PIP_','UV_')):
            env.pop(key)
    # Private interpreter must not import the notebook kernel's Python packages.
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','UV_PROJECT_ENVIRONMENT'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONHASHSEED='42',
               TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               XDG_CACHE_HOME=str(work/'cache'),KERAS_HOME=str(work/'cache/keras'),
               TMPDIR=str(work/'tmp'),PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK='1',
               UV_CACHE_DIR=str(work/'cache/uv'),
               UV_PYTHON_INSTALL_DIR=str(work/'python'),
               UV_PYTHON_BIN_DIR=str(work/'python-bin'))
    return env


def configure_gpu_paths(env, env_python):
    purelib = subprocess.check_output(
        [str(env_python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        text=True, env=env).strip()
    directories = sorted(str(p) for p in Path(purelib).glob('nvidia/*/lib') if p.is_dir())
    previous = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join(directories + ([previous] if previous else []))


def setup(run, work, env_python, env):
    env_dir = work/'env'
    if env_dir.exists() or env_dir.is_symlink():
        if not (env_dir/'pyvenv.cfg').is_file() or not env_python.is_file():
            raise RuntimeError('existing_environment_incomplete_inspect_before_retry:' + str(env_dir))
        check = ('import sys, pathlib, pip; assert sys.version_info[:2] == (3,10); '
                 'assert sys.prefix != sys.base_prefix; '
                 'assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve()')
        if run([str(env_python),'-c',check,str(env_dir)], check=False) != 0:
            raise RuntimeError('existing_environment_failed_preflight_inspect_before_retry:' + str(env_dir))
    else:
        existing = sys.executable if sys.version_info[:2] == (3,10) else shutil.which('python3.10')
        probe = ('import sys, venv, ensurepip; '
                 'assert sys.version_info[:2] == (3,10); print(ensurepip.version())')
        can_create_venv = existing and run([existing,'-c',probe], check=False) == 0
        if can_create_venv:
            # If actual creation now fails (e.g. disk full), preserve the partial
            # environment for inspection rather than installing over it.
            run([existing,'-m','venv',str(env_dir)])
        else:
            bootstrap = work/'bootstrap'
            uv = bootstrap/'bin/uv'
            if not uv.exists():
                run([sys.executable,'-m','pip','install','--no-cache-dir','--only-binary=:all:',
                     '--index-url','https://pypi.org/simple','--target',str(bootstrap),
                     'uv=='+UV_VERSION])
            # Never alter global PATH, drivers or the notebook kernel environment.
            result = subprocess.run([str(uv),'--version'],env=env,capture_output=True,text=True,check=True)
            if result.stdout.split()[:2] != ['uv',UV_VERSION]:
                raise RuntimeError('unexpected_bootstrap_uv_version')
            run([str(uv),'--no-config','venv','--python',MANAGED_PYTHON,'--seed',str(env_dir)])
        run([str(env_python),'-c','import sys; assert sys.version_info[:2] == (3,10)'])
    run([str(env_python),'-m','pip','install','--no-cache-dir','--only-binary=:all:',
         '--index-url','https://pypi.org/simple','-r',str(ROOT/'requirements-linux.in')])
    run([str(env_python),'-m','pip','check'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime',choices=['kaggle','colab'],required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--stage',choices=['diagnose','setup','fetch','prepare-b','audit-b','smoke','pilot','train-a','train-b'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--split-dir',type=Path)
    p.add_argument('--batch-size',type=int,default=128)
    device = p.add_mutually_exclusive_group()
    device.add_argument('--require-gpu',action='store_true')
    device.add_argument('--cpu-only',action='store_true')
    p.add_argument('--include-reference-model',action='store_true',
                   help='Also attempt optional public weights during fetch.')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',args.run_id):
        raise ValueError('invalid_run_id')
    work = args.work.resolve()
    runtime = compute_context(args.runtime,work)
    for dirname in ('tmp','cache','runs'):
        (work/dirname).mkdir(parents=True,exist_ok=True)
    run_dir = work/'runs'/args.run_id
    run_dir.mkdir(exist_ok=False)
    env_python = work/'env/bin/python'
    env = child_environment(work)
    if args.cpu_only:
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    state = dict(stage=args.stage,runtime=runtime,run_id=args.run_id,status='running',
                 device_requested='cpu' if args.cpu_only else ('gpu' if args.require_gpu else 'auto'),
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 source_manifest_sha256=file_sha(ROOT/'configs/upstream.json'),
                 requirements_sha256=file_sha(ROOT/'requirements-linux.in'),
                 launcher_sha256=file_sha(__file__),commands=[],
                 code_sha256={str(f.relative_to(ROOT)):file_sha(f)
                              for base in ('urlcascade','scripts','upstream/Utils','configs')
                              for f in sorted((ROOT/base).glob('*')) if f.is_file()},
                 note='Stage success is not full W1/W2/W3 completion or model evaluation.')
    status_path = run_dir/'status.json'

    def save():
        status_path.write_text(json.dumps(state,indent=2)+'\n')

    def run(argv, check=True):
        index = len(state['commands'])
        record = dict(argv=list(map(str,argv)),log=f'command-{index:02d}.log',status='running',required=check)
        state['commands'].append(record)
        save()
        print('RUN',record['argv'],flush=True)
        with (run_dir/record['log']).open('x') as log:
            try:
                proc = subprocess.Popen(record['argv'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line,end='',flush=True)
                rc = proc.wait()
            except OSError as exc:
                log.write(str(exc)+'\n')
                record['error'] = str(exc)
                rc = 127
        record.update(returncode=rc,status='passed' if rc == 0 else ('failed' if check else 'unavailable'))
        save()
        if rc and check:
            raise RuntimeError('command_failed:'+record['log'])
        return rc

    def ml_check():
        if not env_python.is_file():
            raise RuntimeError('run_setup_first')
        configure_gpu_paths(env, env_python)
        flags = ['--require-gpu'] if args.require_gpu else []
        run([str(env_python),str(ROOT/'scripts/cloud_probe.py'),'--runtime',runtime,
             '--out',str(run_dir/'environment.json'),'--check-ml',*flags])

    save()
    try:
        if args.stage not in ('diagnose', 'fetch'):
            run([sys.executable, str(ROOT / 'scripts/fetch_reference.py')])
        if args.stage == 'diagnose':
            run([sys.executable,str(ROOT/'scripts/cloud_probe.py'),'--runtime',runtime,
                 '--out',str(run_dir/'inventory.json')])
        elif args.stage == 'setup':
            setup(run,work,env_python,env)
            if args.require_gpu:
                run([str(env_python), '-m', 'pip', 'install', '--no-cache-dir',
                     '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                     '--report', str(run_dir/'pip-cuda.json'), '-r',
                     str(ROOT/'requirements-gpu-cu118.in')])
            freeze = subprocess.run([str(env_python),'-m','pip','freeze','--all'],env=env,
                                    capture_output=True,text=True,check=True)
            (run_dir/'requirements-resolved.txt').write_text(freeze.stdout)
            ml_check()
        elif args.stage == 'fetch':
            flags = ['--include-reference-model'] if args.include_reference_model else []
            run([sys.executable,str(ROOT/'scripts/fetch_assets.py'),'--dest',str(work/'data/upstream'),
                 '--report',str(run_dir/'fetch-report.json'),*flags])
            run([sys.executable,str(ROOT/'scripts/verify_assets.py'),'--root',str(work/'data/upstream'),
                 '--report',str(run_dir/'assets.json'),'--required-data-only'])
        elif args.stage == 'prepare-b':
            ml_check()
            run([str(env_python),str(ROOT/'scripts/prepare_b.py'),'--runtime',runtime,
                 '--assets',str(work/'data/upstream'),'--out',str(run_dir/'split')])
        elif args.stage == 'audit-b':
            if args.split_dir is None:
                raise ValueError('explicit_frozen_split_dir_required')
            run([str(env_python),str(ROOT/'scripts/audit_frozen.py'),'--runtime',runtime,
                 '--assets',str(work/'data/upstream'),'--split-dir',str(args.split_dir.resolve()),
                 '--out',str(run_dir/'audit')])
        elif args.stage == 'train-b':
            if args.split_dir is None:
                raise ValueError('explicit_frozen_split_dir_required')
            ml_check()
            run([str(env_python), str(ROOT/'scripts/train_b.py'), '--runtime', runtime,
                 '--split-dir', str(args.split_dir.resolve()), '--out', str(run_dir/'experiment-b')])
        elif args.stage == 'train-a':
            ml_check()
            run([str(env_python), str(ROOT/'scripts/train_a.py'), '--runtime', runtime,
                 '--assets', str(work/'data/upstream'), '--out', str(run_dir/'experiment-a')])
        else:
            if args.split_dir is None:
                raise ValueError('explicit_frozen_split_dir_required')
            ml_check()
            flags = ['--require-gpu'] if args.require_gpu else []
            run([str(env_python),str(ROOT/'scripts/smoke_pilot.py'),'--runtime',runtime,
                 '--split-dir',str(args.split_dir.resolve()),'--out',str(run_dir/'development'),
                 '--stage',args.stage,'--batch-size',str(args.batch_size),*flags])
        state['status'] = 'passed'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        state['output_sha256'] = {str(f.relative_to(run_dir)):file_sha(f)
                                  for f in sorted(run_dir.rglob('*')) if f.is_file() and f != status_path}
        save()
        print(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()
