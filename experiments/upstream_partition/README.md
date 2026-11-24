# 원저자 전처리 기준 악성 URL 탐지 재실험

원저자의 입력·중복 규칙으로 만든 동일한 시험 집합에서 1단계 분기가 악성 미탐 수와 실제 처리 시간에 미치는 영향을 측정한다. 공개 CSV 세 파일과 원저자 코드는 고정 커밋에서 시작한다. CSV는 같은 Git blob의 공개 다운로드 주소와 SHA-256으로 검증한다. URL은 오프라인 문자열로 처리한다.

## 재실행할 조건

| 실험 | 조건과 평가 단위 |
|---|---|
| 원저자 내부 검증·별도 평가 | 앞의 두 파일을 결합해 섞고 전체 열 중복·결측 제거, 라벨 층화 80/20. 학습 집합 어휘와 20회차 GRU를 내부 검증 및 세 번째 파일에 적용. |
| 원저자 최종 모델 | 세 파일을 합쳐 같은 순서로 처리하고 새 어휘와 GRU를 20회 학습. 노트북의 모델별 개별 예시 판정 8회도 다시 실행. |
| 팀 평가 | 정확히 같은 원본 URL의 상충 라벨 전체 제외, 같은 원본 URL·라벨 중복 제거. URL 해시 순서에 기존 층화 또는 등록 도메인 그룹 분할(목표 70/15/15) 적용. 세 역할 내부에서 순서대로 전처리 문자열 첫 행 유지. |
| 학습 조건 | 무작위·도메인 각각 원래 표기와 맨 앞 HTTP(S) 접두어 제거 표기로 새 학습. 대소문자 무시. 네 조건 각각 같은 GRU를 7개 구성에서 공유. |
| 비교 모델 | GRU 단독, LR 또는 HGB와 기본·무작위 5%·경계 5% 분기. LR·HGB 단독과 URL 길이 단독, 접두 표기 유무 단순 규칙. |
| 선택 | 1단계 각 3후보 검증 AP, sigmoid 보정, GRU 20회 최저 검증 손실. 검증에서 추가 미탐 비율이 악성 전체의 1% 이내인 후보 중 호출이 가장 적은 분기 임계값. 동률이면 작은 값. |
| 재검사 | 조기 정상 종료 후보 수에 비율을 곱해 내림. 같은 예산으로 무작위 순위와 분기 경계에 가장 가까운 후보를 비교. 고정 시드 0 및 30개 시드, 임계값 12개 × 비율 0/1/5/10/20%. |
| 보조 분석 | 두 분할에서 접두어 제거 뒤 18개 특징을 재계산해 1단계 재학습(기존 ablation). 별도로 표기 관련 9개 열을 빼고 1단계 재학습. 원래 표기 무작위 분할에서 학습량 10/25/50/75/100%별 검증 AP·로그 손실. |
| 고정 모델 대조 | 원래 표기에서 고정한 모델·어휘·임계값에 같은 시험 행을 원래 표기와 접두어 제거 표기로 입력. 두 입력 재계산, 양쪽에서 행 순서 고정. |
| 재학습 대조 | 두 표기의 전체 집합과 공통 원본 ID 확인. 동일 1,000건에서 7개 구성의 원본 문자열→판정 시간을 예열 후 순서를 바꿔 5회 측정. |
| 과거 실행과의 구분 | 과거 정제 데이터, 행 배정, 모델, 어휘, 보정기, 임계값, 측정 결과를 읽지 않는다. 이전 보고서의 검증 3역할은 최종 설계에서 하나의 검증 집합으로 합쳤다. |

기존 `experiments/clean_rerun/prepare.py`, `refine_inputs.py`, `finalize_data.py`와 `experiments/scheme_ablation/canonical.py`는 서로 다른 URL의 표현·인코딩 충돌을 기준으로 삭제하거나 분할을 다시 만든다. 기존 `experiment.train`에도 최종 입력 충돌 시 중단하는 검사가 있다. 이번 실행에서는 기존의 분할 함수, 1단계 적합 함수, 지표와 분기 함수만 불러온다. 실제 입력의 충돌 수와 분할 간 겹침은 평가 대상의 특성으로 다룬다.

## 실행

Python 3.10과 [실험 환경 버전](../clean_rerun/requirements-experiment.txt)을 사용한다. 아래 명령은 저장소 루트의 PowerShell에서 실행한다. 새 실행 디렉터리를 지정한다. 시간 측정은 모든 학습과 평가가 종료된 뒤 독립적으로 실행한다.

```powershell
$env:URLCASCADE_RUN_DIR = 'C:\url-study\new-run'
$script = 'experiments/upstream_partition'
python "$script/run_study.py" $env:URLCASCADE_RUN_DIR
if ($LASTEXITCODE) { throw "experiment failed" }
```

보고서 작성에는 `python-docx` 1.2.0, `Pillow` 13.0.0, `PyMuPDF` 2.1, 맑은 고딕과 Microsoft Word를 사용했다. 다음 명령은 측정표와 표에 쓰인 원본 수치를 대조하고, Word의 실제 쪽 번호로 목차와 PDF를 만든다. 아래의 `python`은 문서용 패키지가 설치된 환경의 실행 파일이다.

```powershell
python "$script/make_report.py"
if ($LASTEXITCODE) { throw "Word generation failed" }
$reportFile = Get-ChildItem -LiteralPath (Join-Path $env:URLCASCADE_RUN_DIR 'report') -Filter '*.docx' -File
& "$script/render_word.ps1" -Docx $reportFile.FullName -ErrorAction Stop
python "$script/qa_documents.py" $env:URLCASCADE_RUN_DIR
if ($LASTEXITCODE) { throw "document verification failed" }
python "$script/publish_summary.py" $env:URLCASCADE_RUN_DIR
if ($LASTEXITCODE) { throw "Notion summary generation failed" }
```

보고서는 11월 23일 제출본의 표지, 점선 목차, 회색 표 머리글과 가로 테두리를 담은 `nov23-format.docx` 양식을 사용한다. 구성도의 어두운 상자는 팀이 추가한 분기와 재검사를 표시한다.

재실행 시 새 출력 폴더를 사용한다. 여섯 GRU의 입력·어휘·행 목록 해시는 조건별 `training-inputs.json`에 기록되고, 팀 분할별 독립 대조는 `full-input-audit.json`에 남는다. `feature-example.json`에는 원저자 함수와 새 어휘로 계산한 두 표기의 수치·문자 배열이 있다. 두 표기를 합친 14개 구성의 시간 측정 순서는 각 회차의 `pair_order`로 복원할 수 있다. `summary/performance.csv`와 `summary/timing.csv`에는 각각 28개 비교 결과가 있다. `report/document-checks.json`에는 제출본의 장·표·그림 대조 결과가 남는다.

정제 데이터 보관본은 다음 명령으로 만든다. 여섯 ZIP 각각에 역할별 원본 URL·실제 입력 표기·전처리 문자열·라벨·순서·원본 행 ID, 수치·토큰 배열과 어휘가 들어 있다. 생성 후 모든 CSV 행과 텐서를 다시 읽어 대조하고 `manifest.json`에 바이트 수와 SHA-256을 남긴다.

```powershell
python "$script/archive_data.py" $env:URLCASCADE_RUN_DIR (Join-Path $env:URLCASCADE_RUN_DIR 'cohort-archives')
```

각 조건 폴더의 `*-rows.csv.gz`는 순서가 있는 원본 CSV 행 ID와 라벨이다. `results.json`은 시험 성능, `scores.npz`는 행 단위 점수, `fixed-prefix.json`은 고정 모델 표기 비교, `routing.json`은 검증 후보와 선택 결과, `sensitivity.json`은 임계값·비율·시드별 원본이다. `timing.json`에는 1,000개 ID와 실제 실행 순서·소요 시간이 들어 있다. `data.json`에 처리 단계별 행 수와 역할 간 교집합을 기록한다.
