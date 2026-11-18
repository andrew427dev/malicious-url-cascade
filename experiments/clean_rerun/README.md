# 정제 데이터 기반 악성 URL 캐스케이드 재실험

교수님 메일과 과제 안내에 따른 한국어 보고서의 재현 코드다. 진행 상황의 단일 기록은 작업 폴더의 `재실험 작업 기록.md`이며 이 문서는 고정된 실행 방법을 설명한다.

## 확정된 평가 설계

원저자에게 2026-11-11 문의했고 2026-11-16 확인 시 회신이 없었다. 교수님 안내에 따라 세 원파일을 통합·정제하고 학습/검증/시험 목표 비율을 **70/15/15**로 정했다. 최종 데이터는 **717,793행: 정상 376,309 / 악성 341,484**다. 원자료 `type`은 0=악성/1=정상, 평가용 `y_mal`은 1=악성/0=정상이다.

주 분할은 학습 **502,454**, 공통 검증 **107,669**, 시험 **107,670**행이다. 검증 전체의 같은 행을 모델 선택, sigmoid 확률 보정, 분기 임계값 선택에 사용한다. 각각 별도 15%를 만드는 방식이 아니다.

이전 5/5/5 검증 실험의 시험 결과를 이미 확인한 뒤 사용자의 요청으로 변경했다. 후보 격자와 학습·시험 배정은 유지하고 모델은 모두 재학습했다. 동일 검증에서 얻은 보정 후 점수는 독립적인 보정 검증이 아니며 동일 시험 재평가를 처음 보는 맹검 시험으로 주장하지 않는다. 결정과 제한은 `protocol.json`에 고정했다.

도메인 그룹 분할은 별도 강건성 검사다. 학습 **494,984**, 공통 검증 **121,402**, 시험 **101,407**행이다. 그룹의 목표 비율 70/15/15와 실제 행 비율 68.959/16.913/14.128%를 구별한다.

주 비교는 GRU 단독과 LR/트리 각각의 기본·무작위 5%·경계 5% 재검사, 총 7개 구성이다. 두 경량 모델은 동일한 역할의 대안이며 하나의 캐스케이드에 둘 다 직렬로 넣지 않는다. 시험 평가는 모델·보정기·임계값을 고정한 뒤 실행한다. 시험 민감도 분석과 무작위 재검사 30개 seed는 진단이며 설정을 다시 선택하지 않는다.

시간은 정규화된 URL 1,000개 배치에서 실제 선택적 호출을 warm-up 후 5회 측정한다. URL당 상각 시간은 단일 요청 지연이나 p95가 아니다.

## 환경과 원자료

실제 환경: Windows, AMD Ryzen 7 7800X3D, CPU 스레드 1개, Python 3.10.18, TensorFlow 2.12.0, NumPy 1.23.5, pandas 1.5.3, scikit-learn 1.2.2. 별도 Python 3.10 가상환경에 `requirements-experiment.txt`를 설치한다. GPU와 WSL은 사용하지 않았다. 원저자의 recurrent dropout 설정을 유지했다.

[고정 원저자 소스](https://github.com/munna0912/URL_CLASSIFIER/tree/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc)의 세 CSV를 사용한다. `configs/upstream.json`에 고정 다운로드 경로·크기·SHA-256이 있으며 `prepare.py`가 해시를 다시 검사한다. 사전 학습 모델은 사용하지 않는다. 데이터의 URL에는 접속하지 않는다.

저장소 루트에서 참조 코드와 원자료를 준비한다. 기존 파일이 고정 해시와 다르면 다운로드 도구가 중단한다.

```powershell
python scripts/fetch_reference.py
python scripts/fetch_assets.py --dest data
```

제출 ZIP에는 필요한 재사용 코드가 `reference_code`에 포함된다. ZIP에서는 첫 명령을 생략하고 `python scripts/fetch_assets.py --dest reference_code/data`로 원자료를 받는다. ZIP에는 대용량 원자료·정제 배열을 넣지 않았다.

`URLCASCADE_SOURCE`는 `urlcascade/`, `upstream/Utils/Model.py`, `data/dataset/`를 포함하는 참조 루트다. 기본값은 저장소 루트이며 제출 ZIP에서는 자동으로 `reference_code`를 선택한다.

## 새 실행

저장소 또는 ZIP 루트에서 실행한다. 비어 있는 새 출력 경로를 사용한다. 기존 결과를 삭제하거나 덮어쓰지 않는다. 각 명령의 종료 코드가 0인지 확인한 뒤 다음 명령으로 진행한다.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONHASHSEED = '42'
$env:URLCASCADE_RUN_DIR = Join-Path (Get-Location) 'runs/my-new-run'
$experimentPython = '가상환경\Scripts\python.exe'
$code = 'experiments/clean_rerun'
& $experimentPython "$code/prepare.py"
& $experimentPython "$code/refine_inputs.py"
& $experimentPython "$code/finalize_data.py"
& $experimentPython "$code/audit_data.py"
& $experimentPython "$code/capture_environment.py"
& $experimentPython "$code/experiment.py" self-test
& $experimentPython "$code/check_policies.py"
& $experimentPython "$code/experiment.py" random
& $experimentPython "$code/experiment.py" domain
& $experimentPython "$code/learning_curves.py"
& $experimentPython "$code/audit_results.py"
& $experimentPython "$code/diagnostics.py"
```

정제는 `data-before-token-audit` → `data` → `data-final` 순서다. 원문 파싱·정규화, 상충 라벨 격리, 동일 입력 중복 제거, 학습 사전에 따른 실제 token 입력 중복 검사를 거친다. 수동 폴더 이름 변경은 필요 없다. 사전은 각 반복과 최종 단계에서 해당 학습 행으로만 적합한다. 최종 입력은 `data-final`이다.

분할 파일의 `model`, `calibration`, `policy` 키는 이전 배정의 추적을 위해 남아 있다. 현재 실행은 `common.validation_indices()`로 세 집합을 합치고 중복 및 학습·시험 겹침이 없음을 확인한 **전체 검증**만 사용한다.

실행 시작 시 여섯 실행 스크립트와 재사용 코드의 원문·SHA-256을 `source-snapshot.json`에 보존한다. 두 실험 사이에 코드가 바뀌면 중단한다. Git 줄바꿈 변환과 실제 실행 원문을 구별할 수 있다. 제공 결과의 `data-source-snapshot.json`은 공유 검증 실행 전에 동결한 데이터의 원래 정제 코드 증거다.

학습 종료 시 `training-manifest.json`에 가중치·설정·학습 이력 등 11개 파일의 해시를 봉인한다. 일반 평가와 `experiment.py random --resume-evaluation` 또는 `experiment.py domain --resume-evaluation`은 모두 모델 적재 전에 봉인의 파일집합·해시와 20회 학습 기록을 검사한다. Windows 한글 경로에서는 `h5py.File` 핸들로 가중치를 읽는다.

학습 원문은 `source-snapshot.json`, 평가 원문은 `evaluation-source-snapshot.json`으로 각각 보존한다. 제공 결과에는 float32 정상 확률을 악성 확률로 변환할 때 판정 경계를 보존하도록 float64로 먼저 승격한 평가 수정 이력이 포함된다. 수정 전 결과는 각 분할의 `pre-evaluation-fix/`에 보존하며, `evaluation-revision.json`은 기존 완료 해시를 검증한 후 동일 학습 산출물을 봉인한 일회성 이관 기록이다. 이 수정 때문에 모델을 재학습하거나 시험 결과로 설정 격자를 바꾸지 않았다.

## 데이터 사용과 학습 곡선

| 과정 | 주 실험에서 사용하는 행 |
| --- | --- |
| LR 3후보, 트리 3후보 적합 | 학습 502,454행 전체; 후보마다 같은 행 |
| 후보 AP 비교 / GRU 20 epoch checkpoint 선택 | 공통 검증 107,669행 전체 |
| 선택된 LR·트리의 sigmoid 보정 | 같은 공통 검증 107,669행; 학습과 겹침 없음 |
| 분기 임계값 선택 | 같은 공통 검증 107,669행 |
| 최종 정책 지표 | 시험 107,670행 전체 |
| LR·트리 학습량 곡선 | 중첩 학습 50,244 / 125,613 / 251,227 / 376,840 / 502,454행; 매번 같은 검증 107,669행 |
| 시간 측정 | 같은 시험 표본 1,000행; warm-up 1회 및 5회 반복 |

학습 곡선은 선택된 설정으로 학습량을 바꾼 진단이다. 시험 점수는 사용하지 않으며 전체 학습량의 검증 AP가 본 실험의 후보 선택 기록과 일치하는지 검사한다. 같은 행을 여러 후보나 역할에서 썼다고 고유 표본 수를 곱하지 않는다.

## 산출물과 검사

| 경로 | 내용 |
| --- | --- |
| `protocol.json` | 공유 검증 결정, 시험 노출 이력, 고정 후보·선택 규칙 |
| `data-final/manifest.json`, `data-audit.json` | 단계별 개수·클래스·해시, 저장 파일 재검사 |
| `random/`, `domain/` | 분할별 모델·설정·선택·정책·지표·시간·완료 증거 |
| `t1-selection.json`, `gru-selection.json` | 후보 AP와 GRU 20회 학습 이력 |
| `policy-frozen.json` | 시험 평가 전에 고정한 임계값·모델 해시 |
| `results.json`, `test-scores.npz` | 시험 지표·민감도·무작위 30개 seed 및 원점수 |
| `learning-curves.json`, `learning-curve-indices.npz` | 부분집합·클래스 수·학습/검증 곡선 |
| `result-audit.json` | 혼동행렬, 재검사 동일 예산, 실제 호출, 시간, 곡선의 별도 재계산 |
| `diagnostics.json` | 고정 ID로 선택한 오류·복구 사례; 제공 결과의 평가 수정 전후 불변성 검사 |
| `previous-results/` | 이전 5/5/5 결과와 설정 이력; 새 학습 입력이 아님 |

`tests/test_clean_rerun_contract.py`는 검증 합집합의 학습·시험 겹침 거절과 실행 원문 변조 거절을 검사한다. `experiment.py self-test`는 작은 경계 사례를, `check_policies.py`는 분기·재검사 예산·복구 계산의 48개 합성 사례를 확인한다. 단위 검사와 실제 데이터·시험 지표 재계산은 구별한다.

`diagnostics.py`는 결과 감사를 통과한 점수에서 조기 종료 미탐, 경계 재검사 복구·추가 오탐, GRU 자체 미탐을 추출한다. 각 유형의 가장 작은 고정 ID를 설명용 예시로 쓰며 원문 URL은 결과에 쓰지 않는다. 제공 결과의 평가 수정 이력이 있으면 원래 완료 해시, 동일 가중치·시험 행·경량 점수, float32→float64 계산 관계도 검사한다. 오류 사례로 설정을 재선택하지 않는다.

## 한국어 보고서와 제출 ZIP

보고서 생성에는 ReportLab 4.4.9와 Windows의 `malgun.ttf`, `malgunbd.ttf`가 필요하다. 다른 OS에서는 한글 TrueType 글꼴 경로를 바꾼다. `build_report.py --draft`는 학습 완료 전 별도 초안을 만든다. 최종 이름의 보고서는 두 실행과 데이터·결과 감사가 PASS일 때 생성한다.

```powershell
& $reportPython 'experiments/clean_rerun/build_report.py'
& $experimentPython 'experiments/clean_rerun/package_results.py'
```

`$reportPython`은 ReportLab을 설치한 Python 실행 파일이다. 제공 보고서에는 이전 비교·환경·공식 템플릿 출처도 들어 있으므로 이 자료를 포함한 제공 결과 폴더에서 다시 생성한다. 새 독립 실험에는 해당 증거를 별도로 준비해야 한다. 패키징은 모델 가중치·원점수·보고서·코드와 파일별 해시를 ZIP에 넣고 압축 및 해시를 검사한다.

과제 PDF의 필수 절과 최소 8쪽을 우선하고, [CMU ML 프로젝트 지침](https://www.cs.cmu.edu/~lwehbe/10701_S20/files/S20_10701_Projects_v3.pdf)과 [NeurIPS 2025 공식 서식](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip)의 단일단·표·그림 구성을 참고했다. 한국어 가독성을 위해 본문 10pt/행간 15pt로 조정했다. 학회 투고를 의미하지 않는다.

정제로 해결한 범위는 검사 가능한 URL·모델 입력 중복과 상충 라벨이다. 남은 모든 라벨을 외부 정답으로 검증했다는 뜻은 아니다. 원저자와 다른 데이터의 성능을 동등 조건 비교로 주장하지 않는다. 분할별 모델 학습 seed는 하나이며 30개 seed 반복은 무작위 재검사 선택에만 해당한다.
