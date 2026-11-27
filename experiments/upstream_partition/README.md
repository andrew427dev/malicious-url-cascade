# 최종 실험 v1과 이전 실험 실행 안내

최종 실험 v1을 새로 학습·평가하려면 아래 [최종 실험 v1 전체 재실행](#최종-실험-v1-전체-재실행)의 `rerun_v1.py`를 사용한다. 저장된 최초 실행 결과를 추가 분석하는 방법은 [v2 분석 안내](../final_report_v2/README.md)에 있다. 최신 보고서는 [Notion의 v3](https://app.notion.com/p/424aebb23ee0809a8cbac7a4fa43839c)이며, v3 생성기는 이 저장소에 포함되어 있지 않다.

| 목적 | 필요한 자료와 진입점 |
|---|---|
| 최초 결과 재검산·추가 분석 | 완전한 최초 v1 실행 폴더와 v2 분석 코드. [보관 묶음의 범위](#실행-자료-보관-위치)를 먼저 확인한다. |
| 새 학습·평가 | 아직 없는 새 폴더에 `rerun_v1.py` 실행. 결과는 새 실행의 `results/`와 검증 기록으로 확인한다. |
| 최초 v1 보고서·공유 자료 재생성 | [최초 실행의 문서와 공유 자료](#최초-실행의-문서와-공유-자료). 최초 실행의 과정 설명이 고정된 문서 생성 코드는 새 실행용으로 쓰지 않는다. |
| 11월 24일 기존 실험 확인 | 아래 `run_study.py` 설명과 [당시 병합본](https://github.com/andrew427dev/malicious-url-cascade/tree/5fc1b2d67f2a99292ba3b4fb4c4f59bfa577166a). 최종 v1의 정제 비교·30회 짝측정과 구분한다. |

## 2026-11-24 기존 실험 조건

원저자의 입력·중복 규칙으로 만든 동일한 시험 집합에서 1단계 분기가 악성 미탐 수와 실제 처리 시간에 미치는 영향을 측정한다. 공개 CSV 세 파일과 원저자 코드는 고정 커밋에서 시작한다. CSV는 같은 Git blob의 공개 다운로드 주소와 SHA-256으로 검증한다. URL은 오프라인 문자열로 처리한다.

다음 조건과 `run_study.py` 실행 안내는 11월 24일 기존 실험의 보존 기록이다.

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

## 2026-11-24 기존 실험 실행

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

## 최종 실험 v1 전체 재실행

최종 실험 보고서 v1은 2026-11-24에 시작한 실행에서 위 실행을 확장한 코드로 만들었다. 정제 전에 모든 원본 URL의 역할을 배정하고(`assign_roles.py`), 같은 배정에서 두 정제 절차를 비교한다(`cohorts.py`의 `reference`는 기본 정제, `team`은 URL 정규화와 모델 입력 단위 충돌·중복 제거). 정제 2 × 분할 2(URL 무작위·등록 도메인) × 표기 2(원래·접두어 제거)의 비교 조건 8개에서 GRU를 하나씩, 원저자 재현에서 GRU 2개를 학습한다.

`rerun_v1.py`는 이 과정을 아직 없는 새 폴더에서 처음부터 다시 하는 진입점이다. 최초 실행은 원저자 재현 모델 두 개(`author-validation`, `author-final`)를 병행 학습했지만, `rerun_v1.py`는 `author-validation`의 학습과 별도 평가를 마친 뒤 `author-final`을 학습한다.

역할 배정표의 해시는 두 실행 모두 같은 방식으로 남는다.

- `assign_roles.py`가 배정 직후 `source-manifest.json`에 배정표 CSV의 해시를 기록한다.
- 학습 직전 기록(`training-inputs.json`)에는 배정표와 `assign_roles.py`·`cohorts.py`의 해시를 넣지 않는다.
- 마무리 단계의 `review_evidence_v1.py`가 세 파일의 해시를 다시 계산해 `results/review-evidence.json`에 남기고, 배정표 해시가 `source-manifest.json`의 기록과 같은지 확인한다.

저장소 루트에서 다음 명령을 실행한다.

```powershell
python experiments/upstream_partition/rerun_v1.py 'D:\runs\final-v1-rerun'
if ($LASTEXITCODE) { throw "v1 rerun failed" }
```

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 준비 | `fetch`, `inputs parity`, `assign_roles` | 공개 CSV·노트북 해시 대조, 원저자 함수와 입력 동등성 검사, 정제 전 역할 배정 |
| 원저자 재현 | `inputs`, `train_gru` (`author-validation`, `author-final`), `score_v1 author-validation` | 앞 두 파일 80/20 학습·내부 검증과 세 파일 전체 학습. 세 번째 파일의 별도 평가는 `author-validation` 모델로만 한다 |
| 조건별 학습 | `run_comparison` → `cohorts`, `train_gru`, `fit_stages`, `evaluate --freeze`, `score_v1`, `provenance example` | 조건마다 정제·GRU 학습, 1단계 선택·보정, 검증 임계값 고정, 시험 점수와 고정 모델 접두어 대조. 조건별 학습은 최대 세 개를 함께 돌린다 |
| 마무리 | `finish_v1` → `paired_time`, `aggregate`, `audit_v1`, `review_evidence_v1`, `demo_v1`, `demo_screen_v1` | 1,000건 × 30회 짝측정, 결과표·구간 집계, 독립 감사(`results/verification.json`), 사후 증거 확인, 시연 사례 |

새 실행의 결과는 `results/`의 표와 `results/verification.json`으로 확인한다.

### 최초 실행의 문서와 공유 자료

다음 스크립트는 최초 실행 폴더에서 보고서 v1과 공유 자료를 만든 코드다. `report_v1.py`·`publish_v1.py`의 실행 과정 문단은 최초 실행의 기록(원저자 모델 병행 학습, 학습 뒤 해시 확인)으로 고정돼 있다. 그래서 `rerun_v1.py`로 만든 새 실행의 문서 생성에는 쓰지 않는다.

| 번호 | 스크립트 | 읽는 것 → 만드는 것 |
|---|---|---|
| 1 | `archive_inputs_v1.py` | 조건별 토큰·수치 배열, 행 ID·라벨, 어휘 → `share/` 아래 학습·평가 입력 ZIP 3개(`share` 폴더가 있어야 하고 같은 이름이 있으면 실패) |
| 2 | `share_models_v1.py` | `results/verification.json`(PASS), 모델·보정기·분기 기록, 행별 점수 → `share/` 아래 모델 ZIP과 점수 ZIP |
| 3 | `figure_v1.py` | `results/threshold-decisions.csv` → `report/`의 분할별 그림 |
| 4 | `report_v1.py` | 결과표와 기록 → `report/악성 URL 탐지 최종 실험 보고서 v1.docx`, `report/cascade.png` |
| 5 | `publish_v1.py` | 결과표, 실행 폴더의 `code`(저장소와 같은 구조의 코드 사본), Word → 결과·재실행 자료 ZIP, `results/재실행 안내 v1.md`, `layout-qa/rate-tables.html`(`layout-qa` 폴더가 있어야 함), `report/notion-body.html`, `report/notion-content.json` |
| 6 | `render_word.ps1 -Docx <Word>` | Word를 다시 저장하며 필드·쪽 번호 갱신 → 같은 폴더의 PDF |
| 7 | `check_report_v1.py` | Word, PDF, `report/notion-content.json` → `layout-qa/`의 쪽 이미지와 대조 결과 |

3~7번은 최초 실행 폴더의 `build-when-ready.ps1`이 실행하는 순서다. 1·2번은 이 스크립트 밖에서 따로 실행했다.

### 실행 당시 코드와 다른 점

이 폴더의 파일은 최초 실행 폴더의 코드 사본(`code/experiments/upstream_partition`)과 이 README를 빼고 줄바꿈 외에 같다. 다른 파일은 `provenance.py` 하나다. Drive `04`의 코드 묶음은 이 가운데 최종 v1 실행에 쓰는 파일만 골라 담았고(`run_study.py` 등 6개는 없다), 담긴 파일은 실행 사본과 같다.

- 저장소의 `provenance.py`는 이 저장소의 기존 판(`5fc1b2d`)이다. 최초 실행의 모델 10개가 실제로 남긴 학습 직전 기록과 같은 항목을 기록한다(비교 조건 22개, `author-validation` 22개, `author-final` 16개).
- 코드 묶음의 판은 최초 학습이 끝난 뒤, 앞으로의 재실행에서 배정표와 두 정제 코드의 해시를 학습 직전 기록에 넣도록 고친 판이다. 이 판에는 두 가지 문제가 있다.
  - 배정표가 없는 위의 `run_study.py` 흐름을 학습 직전에 멈춘다.
  - 이 판으로 새로 학습한 비교 모델은 학습 직전 기록에 키 25개를 남기는데, 저장소판 `verify_seal`로는 읽을 수 없다. 저장소판 v2 분석에 쓸 새 실행은 저장소판 `rerun_v1.py`로 만든다.
- 보고서의 수치는 바뀌지 않는다.

추가 분석과 보고서 v2는 [`../final_report_v2`](../final_report_v2/README.md)에 있다.

### 실행 자료 보관 위치

최초 실행의 자료는 팀 Google Drive의 '최종 재실험 v1' 폴더에 SHA-256 목록과 함께 보관했다. 폴더는 링크 공개가 아니며 공유받은 계정만 열 수 있다.

| 하위 폴더 | 주요 파일 |
|---|---|
| 01 보고서와 시연 | `악성 URL 탐지 최종 실험 보고서 v1.docx`·`.pdf`, `시연 자료 v1.md`, `악성 URL 탐지 시연 v1.html`, `시연 사례 v1.csv` |
| 02 결과표와 측정 원본 | `결과표·측정 원본·재실행 자료 v1.zip`(결과표 CSV, 조건별 기록, 30회 시간 측정 원본, 코드 사본), `시험 URL별 정답·예측 점수 v1.zip`(행별 점수 14개), `정제·접두어 시간 짝비교 v1.csv` 등 CSV |
| 03 데이터와 학습 모델 | `공개 원본과 정제·분할 기록 v1.zip`, `원저자 재현 학습·평가 입력 v1.zip`, `기본 표기 학습·평가 입력 v1.zip`, `접두어 제거 학습·평가 입력 v1.zip`(토큰·수치 배열, 행 ID·라벨, 어휘), `새 학습 모델·어휘·보정기·분기 기록 v1.zip` |
| 04 재실행 코드 | `재실행 코드와 안내 v1.zip`, `재실행 안내 v1.md` |

보관본 사용 시 유의할 점은 다음과 같다.

- `02`의 통합 ZIP과 `04`의 두 파일에 있는 안내문(`README.md`, `재실행 안내 v1.md`)은 최초 실행 당시 안내다. 그 안의 "보고서 재생성" 지시(`figure_v1.py` → `report_v1.py` → `render_word.ps1`)는 새 실행에는 적용하지 않는다(위 "최초 실행의 문서와 공유 자료").
- `04`의 코드 묶음은 `provenance.py`가 저장소와 다르다. 그 코드로 새로 학습한 폴더는 저장소판 v2 분석에 쓸 수 없다(위 "실행 당시 코드와 다른 점").
- 이 묶음에는 다음 파일이 없다. 그래서 풀기만 해서는 실행 폴더가 완성되지 않으며, 실행 폴더가 필요하면 `rerun_v1.py`로 새로 만든다.
  - pickle 파일(`*.pkl`)
  - 일부 조건별 기록(`controls-final.json`, `sensitivity.json`, `fixed-prefix.json` 등)
- 다시 학습한 모델의 수치가 보고서와 같다는 보장은 없다.
