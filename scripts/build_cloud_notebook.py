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
    md('''# 악성 URL 캐스케이드 · 데이터 준비와 파일럿\n
Kaggle 또는 Colab에서 새 세션을 열고 필요한 단계를 선택합니다.
이 파일은 실제 실행 결과를 포함하지 않습니다. 기본 단계는 **diagnose**로, 설치·다운로드·학습을 하지 않습니다.
실행 순서는 setup → fetch → prepare-b → audit-b입니다.

- Python 3.10 / TF 2.12 고정 환경을 별도 subprocess로 사용합니다. Notebook 커널의 패키지를 교체하지 않습니다.
- setup: 전용 venv. Python 3.10이 없으면 공식 PyPI의 uv 0.9.30 및 Astral 관리 Python 3.10.19를 작업 폴더에 설치합니다.
- fetch: 고정 GitHub CSV 3개를 받고 SHA-256·크기를 검사합니다. 공개 모델은 선택적 참고 파일이며 실패해도 필수 CSV 확보를 막지 않습니다. 데이터 안 URL은 방문하지 않습니다.
- prepare-b: B 분할을 생성합니다. audit-b에서 고정 파일·도메인 교집합·문자열 충돌과 외부 겹침을 검사합니다.
- smoke/pilot: 명시한 동결 B 분할의 학습 행만 사용합니다. 최종 A/B 학습 실행기가 아닙니다.
- GPU 필요 여부·계정 할당량·Notebook 버전 출력 보존은 별도 확인해야 합니다.
''')
    code('''# 단계별 설정: 한 번에 한 단계만 실행하며 매번 새 RUN_ID를 사용합니다.
STAGE = "diagnose"  # diagnose / setup / fetch / prepare-b / audit-b / smoke / pilot / train-a / train-b
REQUIRE_GPU = False  # GPU를 선택한 setup/smoke/pilot 검증 때만 True
CPU_ONLY = False  # 장치 비교 또는 CPU 본 학습에서 True
INCLUDE_REFERENCE_MODEL = False  # fetch에서 공개 모델 참고용 확보를 시도할 때만 True
SPLIT_DIR = None  # audit-b/smoke/pilot에서 실제 동결 split 디렉터리 절대경로 지정
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
    code('''# 코드만 포함한 고정 묶음. 전체 묶음과 개별 파일의 무결성을 검사합니다.
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
print('단계 출력 위치:',status_file.parent)
print('완료 결과는 Notebook 버전 출력과 별도 보관본에 저장한 뒤 새 세션에서 확인해야 합니다.')
''')
    md('''## 이후 단계의 완료 기준

setup의 CPU 검사 통과는 GPU 학습 성공을 뜻하지 않습니다. fetch의 파일 검증 통과는 데이터 감사·분할 동결을 뜻하지 않습니다.
prepare-b 결과의 `remaining_audit`를 검토하고, 동결 파일 해시를 확인한 뒤 B train에서만 smoke/pilot를 수행합니다.
원본 GRU 설정, A/B 가중치·어휘 분리, T2 정상 점수 0.5 경계와 `p_mal < tau` 규칙을 유지합니다.
중단 체크포인트의 정확한 확률적 학습 재개는 아직 구현되지 않았으므로 개발 실행은 다시 시작하거나 차이를 명시합니다.
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
