"""Bundle the experiment runners. Reference modules are fetched at runtime."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def payload_files():
    files = [ROOT/'requirements-linux.in', ROOT/'requirements-gpu-cu118.in', ROOT/'configs/upstream.json']
    for folder in ('urlcascade','scripts'):
        files.extend(sorted((ROOT/folder).glob('*.py')))
    return sorted(set(files))


def make_bundle():
    buffer = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in payload_files():
            name = str(file.relative_to(ROOT))
            data = file.read_bytes()
            info = zipfile.ZipInfo(name,date_time=(2026,11,7,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
            hashes[name] = hashlib.sha256(data).hexdigest()
    return buffer.getvalue(),hashes


def main():
    bundle,hashes = make_bundle()
    digest = hashlib.sha256(bundle).hexdigest()
    payload = base64.b64encode(bundle).decode()
    cells = []
    def md(s):
        cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
    md('''# Malicious URL cascade: data preparation and pilot runs\n
Open a fresh Kaggle or Colab session and choose a stage. The default stage, **diagnose**, checks the environment. The preparation stages run in this order: setup, fetch, prepare-b, then audit-b.

- The runner installs Python 3.10 and TensorFlow 2.12 in a separate environment for child processes.
- setup creates the virtual environment. If Python 3.10 is unavailable, it obtains uv 0.9.30 from PyPI and Astral's managed Python 3.10.19.
- fetch retrieves the three pinned CSVs and verifies their sizes and SHA-256 digests. Reference weights are optional. Dataset URLs are processed offline.
- prepare-b creates the B split. audit-b checks file integrity, domain overlap, representation collisions and overlap with the external source.
- smoke and pilot sample only frozen B training rows. Use the separate A and B notebooks for full training.
- Check the selected accelerator, remaining account quota and saved notebook outputs in Kaggle.
''')
    code('''# Select one stage per run. Each execution creates a fresh RUN_ID.
STAGE = "diagnose"  # diagnose / setup / fetch / prepare-b / audit-b / smoke / pilot / train-a / train-b
REQUIRE_GPU = False  # Enable for GPU setup, smoke or pilot checks.
CPU_ONLY = False  # Enable for CPU training or the CPU side of a device comparison.
INCLUDE_REFERENCE_MODEL = False  # Enable to fetch the optional upstream reference weights.
SPLIT_DIR = None  # Set the absolute frozen split path for audit-b, smoke or pilot.
BATCH_SIZE = 128
''')
    code('''import base64, hashlib, io, json, os, sys, subprocess, zipfile
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
if Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir():
    RUNTIME, BASE = 'kaggle', Path('/kaggle/working')
elif Path('/content').is_dir() and (os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION')):
    RUNTIME, BASE = 'colab', Path('/content')
else:
    raise RuntimeError('Kaggle/Colab execution context not verified; no installation or training started')
WORK = BASE / 'hufs-compsec-url-cascade'
WORK.mkdir(exist_ok=True)
RUN_ID = STAGE + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
print({'runtime':RUNTIME,'work':str(WORK),'stage':STAGE,'run_id':RUN_ID})
''')
    code('''# Verify the pinned code bundle and every file inside it.
PAYLOAD_SHA256 = '''+repr(digest)+'''
PAYLOAD = '''+repr(payload)+'''
EXPECTED_FILES = '''+repr(hashes)+'''
raw = base64.b64decode(PAYLOAD, validate=True)
assert hashlib.sha256(raw).hexdigest() == PAYLOAD_SHA256, 'payload hash mismatch'
CODE = WORK / ('code-' + PAYLOAD_SHA256[:16])
CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert set(archive.namelist()) == set(EXPECTED_FILES), 'unexpected payload members'
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        assert not path.is_absolute() and '..' not in path.parts, 'unsafe archive path'
        target = CODE / path
        assert target.resolve().is_relative_to(CODE.resolve()), 'unsafe output path'
        data = archive.read(member)
        assert hashlib.sha256(data).hexdigest() == EXPECTED_FILES[member.filename]
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_FILES[member.filename], 'existing code mismatch'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
print({'code_files':len(EXPECTED_FILES),'code_sha256':PAYLOAD_SHA256})
''')
    code('''argv = [sys.executable,str(CODE/'scripts/cloud_run.py'),'--runtime',RUNTIME,
        '--work',str(WORK),'--stage',STAGE,'--run-id',RUN_ID,'--batch-size',str(BATCH_SIZE)]
if REQUIRE_GPU:
    argv.append('--require-gpu')
if CPU_ONLY:
    argv.append('--cpu-only')
if INCLUDE_REFERENCE_MODEL:
    argv.append('--include-reference-model')
if SPLIT_DIR is not None:
    argv += ['--split-dir',str(SPLIT_DIR)]
subprocess.run(argv,check=True)
''')
    code('''status_file = WORK/'runs'/RUN_ID/'status.json'
status = json.loads(status_file.read_text())
print(json.dumps(status,indent=2))
assert status['status'] == 'passed', 'stage not complete'
print('Stage output directory:',status_file.parent)
print('Save the completed outputs with the notebook version, keep a backup and verify restoration in a fresh session.')
''')
    md('''## Completion checks

Verify GPU operation in a GPU session when required. Review the data audit after source retrieval.
Review `remaining_audit` in the prepare-b result, confirm the split hashes, then run smoke and pilot on B training rows only.
Keep A and B weights and vocabularies separate. Preserve the upstream GRU configuration, the T2 benign-score boundary at 0.5 and the early-exit rule `p_mal < tau`.
If training stops, restart the development run or document the changed conditions. The runner does not restore the exact stochastic training state.
''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'hufs':{'stage':'W1-W3 preparation','payload_sha256':digest,
        'source_commit':'835bd5e1710531407c0b5ca09da3d7c3de8d1bbc','executed':False}})
    for i,cell in enumerate(cells):
        cell['id'] = f'hufs-{i:02d}'
    target = ROOT/'notebooks/HUFS_CompSec_Cloud.ipynb'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    # Keep the public A/B launchers in step with the common bootstrap.
    for experiment in ('A', 'B'):
        runner = ROOT / 'notebooks' / f'HUFS_CompSec_Experiment_{experiment}.ipynb'
        version = json.loads(runner.read_text())
        version['cells'][3]['source'] = cells[3]['source']
        version['metadata']['hufs']['payload_sha256'] = digest
        runner.write_text(json.dumps(version, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({'notebook':str(target),'payload_files':len(hashes),'payload_sha256':digest,
                      'notebook_bytes':target.stat().st_size}))


if __name__ == '__main__':
    main()
