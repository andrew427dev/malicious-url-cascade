# 최종 실험 보고서 v2 분석 코드

이 폴더의 실험 0~6은 v1 최종 실행 폴더(`experiments/upstream_partition/rerun_v1.py` 출력)의 고정 모델·점수·입력을 **읽기만** 하는 추가 분석이다. 재학습은 없다. 실험 5만 고정 GRU로 검증 집합 점수를 새로 계산한다. 결과는 v2 폴더의 `results/`에, 보고서는 `report/`에 쓴다.

## 폴더 지정

| 환경 변수 | 뜻 |
|---|---|
| `URLCASCADE_V1_DIR` | v1 최종 실행 폴더 전체(조건별 `*.pkl`·점수·모델·기록 포함, 루트에 `source-manifest.json`과 `results/verification.json`이 모두 있어야 함). 최초 v1 실행 폴더나 저장소판 `rerun_v1.py`로 만든 폴더를 쓴다. 저장소판 `rerun_v1.py`로 만든 새 폴더는 저장소 배치에서 분석한다. 원래 `final-v2/code` 배치에서 분석하려면 입력 v1 폴더에 `code/` 사본이 있어야 한다(그 배치는 v1 모듈을 `<v1>/code`에서 가져온다). Drive 보관 묶음만으로는 부족하다([보관 위치](../upstream_partition/README.md#실행-자료-보관-위치)) |
| `URLCASCADE_V2_DIR` | v2 결과(`results/`, `report/`)를 쓸 폴더 |

저장소에서 실행할 때는 두 변수를 모두 지정한다. 저장소 밖의 원래 출력 배치(`final-v1`과 `final-v2/code`가 나란히 있는 폴더)에서 두 변수를 모두 지정하지 않았을 때만 `final-v2/code`의 위치로 두 폴더를 정한다. 폴더 해석과 보호 검사는 `v2paths.py`가 맡는다.

`v2common.py`는 `URLCASCADE_RUN_DIR`를 v1 폴더로 두고 v1 코드 경로를 `sys.path`에 넣는다. 각 실험 스크립트가 v1 모듈(`evaluate`, `inputs`, `experiment`, `aggregate`, `cohorts`)을 가져온다. 파일을 쓰는 v1 함수(`evaluate.team`·`freeze`, `paired_time.main`, `aggregate.main`, 보고서 생성 함수)는 호출하지 않는다. `load_test`는 `test.pkl`·`scores.npz`·`test-rows.csv.gz`의 행 순서와 라벨이 같은지 단언한다.

**v1 보호:** 이 README의 명령이 쓰는 폴더가 루트에 `source-manifest.json`이 있는 폴더(v1 실행 폴더와 그 사본) 안이면, 그 폴더에 아무것도 만들거나 지우지 않고 `refused[v1_protected]`로 멈춘다. 대상 폴더는 `results/`, `report/`, `e6_rate_timing.py dryrun`의 출력 폴더, `render_pages.py`의 `layout-qa`다. 다음 경우는 이 검사가 막지 않는다.

- 표지가 없는 v1 폴더: `assign_roles.py` 전에 멈춘 새 실행 폴더, `source-manifest.json`이 빠진 부분 사본. 이런 폴더를 출력 폴더로 쓰지 않는다.

- 경로를 인자로 받는 함수를 직접 호출하는 경우: `diagram_v2.draw`, `report_base.figure_thresholds`·`figure_rates`, `e6_rate_timing.measure`·`summarize`, `write_json`. 이때는 호출자가 경로를 책임진다.
- 이미 있는 출력 파일이 v1 파일을 가리키는 링크인 경우. v1 실행 폴더나 그 사본(하드링크 사본 포함)을 v2 출력 폴더로 쓰지 않는다.

## 실행 환경

- **실험(e0~e6):** Python 3.10과 [실험 환경 버전](../clean_rerun/requirements-experiment.txt)(TensorFlow 2.12, scikit-learn 1.2.2). v1 폴더에 바이트코드 캐시가 생기지 않도록 `python -B`와 `PYTHONDONTWRITEBYTECODE=1`로 실행한다.
- **보고서(`report_v2.py`, `notion_v2.py`, `render_pages.py`):** python-docx, Pillow, matplotlib, pandas, numpy, PyMuPDF. TensorFlow는 필요 없다. PDF 변환에는 Microsoft Word와 맑은 고딕을 쓴다.
- **줄바꿈 주의:** GRU를 불러올 때 `provenance.verify_seal`이 학습 당시 코드 파일의 SHA-256을 대조한다. v1은 Windows의 CRLF 체크아웃(`core.autocrlf=true`)에서 기록했으므로, 다른 줄바꿈으로 받은 저장소에서는 이 검사가 실패한다.
- **시간 측정:** 실험 6은 같은 장비(v1과 같은 CPU)에서 다른 무거운 작업이 없을 때 실행한다.

## 실행 순서

저장소의 이 폴더에서 PowerShell로 실행한다.

```powershell
$env:URLCASCADE_V1_DIR = 'D:\runs\final-v1'
$env:URLCASCADE_V2_DIR = 'D:\runs\final-v2'
$env:PYTHONUTF8 = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($s in 'e0_invalid_reasons','e1_baselines','e2_failures','e3_cleaning','e4_prefix','e5_equal_fpr') {
    python -B "$s.py"; if ($LASTEXITCODE) { throw $s }
}
python -B e6_rate_timing.py measure;   if ($LASTEXITCODE) { throw 'e6 measure' }
python -B e6_rate_timing.py summarize; if ($LASTEXITCODE) { throw 'e6 summarize' }
```

보고서는 문서용 환경에서 만든다. 아래 Word 변환은 `V2_DRAFT`·`V2_E6_DIR`를 지정하지 않은 정식 생성이 성공한 뒤에만 한다.

```powershell
python -B report_v2.py
if ($LASTEXITCODE) { throw 'report_v2' }
& ..\upstream_partition\render_word.ps1 -Docx (Join-Path $env:URLCASCADE_V2_DIR 'report\악성 URL 탐지 최종 실험 보고서 v2.docx')
python -B render_pages.py (Join-Path $env:URLCASCADE_V2_DIR 'report\악성 URL 탐지 최종 실험 보고서 v2.pdf')
if ($LASTEXITCODE) { throw 'render_pages' }
python -B notion_v2.py
if ($LASTEXITCODE) { throw 'notion_v2' }
```

`render_pages.py`는 PDF를 먼저 연 뒤 `report/layout-qa`의 기존 `page-*.png`를 지우고 각 쪽을 `page-NN.png`로 저장한다. 환경 변수가 필요 없다. `notion_v2.py`는 Notion 게시용 요약을 `report/notion-v2.md`·`notion-v2.html`로 쓴다. `e6_rate_timing.py dryrun <폴더>`는 측정기를 2회만 돌려 지정 폴더에 쓰는 점검용이다.

## 실험과 결과 파일

| 실험 | 스크립트 | 결과 파일 | 실행 중 대조 |
|---|---|---|---|
| 0 잘못된 URL 제외 사유 | `e0_invalid_reasons.py` | `e0-invalid-reasons.csv`, `e0-checks.json` | 사유별 합계가 팀 정제 `invalid_url` 단계의 행 수·악성 수와 일치 |
| 1 단순 기준·단독 모델 | `e1_baselines.py` | `e1-baselines.csv`, `e1-checks.json` | URL 길이·접두어 규칙은 `controls-final.json`, 1단계 단독은 `results.json`, GRU는 `main-28.csv`와 혼동행렬·AP 일치 |
| 2 추가 미탐 유형 | `e2_failures.py` | `e2-miss-summary.csv`, `e2-miss-breakdown.csv`, `e2-miss-cases.csv`, `e2-checks.json` | 12개 캐스케이드의 추가 미탐 수가 `main-28.csv`와 일치. 분류마다 수준 합계가 GRU 탐지 악성·추가 미탐 합계와 일치 |
| 3 정제 수준 비교 | `e3_cleaning.py` | `e3-overlap-counts.csv`, `e3-common-subsets.csv`, `e3-representation.csv`, `e3-representation-columns.csv`, `e3-checks.json` | 입력 동일 행 5,526/1,263(URL 무작위), 31/11(도메인)이 `review-evidence.json`과 일치. 팀 정제는 0 |
| 4 접두어 판정 변화 | `e4_prefix.py` | `e4-prefix-flips.csv`, `e4-prefix-overall.csv`, `e4-prefix-share.csv`, `e4-checks.json` | 제거 표기의 혼동행렬·호출 수가 `fixed-prefix.json`과, 접두어별 시험 행 수가 `controls-final.json`과 일치 |
| 5 같은 오탐률 비교 | `e5_equal_fpr.py` | `e5-equal-fpr.csv`, `e5-validation-scores-*.npz`, `e5-checks.json` | 새 GRU 검증 점수로 계산한 기본 분기 grid(2분할×2모델×임계값 12개)가 `routing.json`과 일치 |
| 6 재검사율 시간 | `e6_rate_timing.py measure` → `summarize` | `e6-timing-{split}.csv`, `e6-timing-meta.json`, `e6-rate-summary.csv`, `e6-v1-comparison.csv`, `e6-checks.json` | 매 실행에서 수치·토큰·경로·판정을 저장 점수와 단언. 전체 시험 탐지값이 `sensitivity.csv`와, 0%·5%의 배치 호출 수가 `timing-summary.csv`와 일치 |

## 세부 규칙

- **실험 2 TLD 묶음:** 등록 도메인의 마지막 이름표를 TLD로 본다. IP 주소와 점 없는 이름은 따로 둔다. 분할마다 GRU가 탐지한 시험 악성 수가 많은 TLD 6개(동점은 이름순)를 고정하고 나머지는 '기타'로 묶는다. 미탐을 보기 전에 정하는 기준이며 모든 구성이 같은 행을 쓴다.
- **실험 2 경로·쿼리:** 정규화 URL을 `urlsplit`으로 나눠 경로나 쿼리가 비어 있지 않으면 '경로·쿼리 있음'이다.
- **실험 2 대표 사례:** 주소는 무력화한 뒤 말줄임표를 포함해 70자 이내로 자르고 `[.]`를 가르지 않는다.
- **실험 2 길이 사분위:** 각 조건 학습 URL(원본 문자열) 길이의 사분위수로 시험 URL을 나눈다. 대표 사례는 경계 재검사 두 구성마다 추가 미탐 행을 원본 행 ID의 SHA-256 순으로 3개씩 고른다.
- **실험 3 입력 동일:** v1 `review_evidence_v1.signatures`(토큰 50칸 int32 바이트 + 수치 18개 float32 바이트의 SHA-256)와 같은 키를 쓴다.
- **실험 3 '앞 52자+수치':** v1 `cohorts.build`의 키(모델 문자열 앞 52자와 수치 특징)를 쓴다. v1은 수치 목록을 `json.dumps`로 해시했고 여기서는 float32 바이트를 쓴다. 수치가 모두 정수임을 단언하므로 같은 값은 같은 키로 묶인다.
- **실험 5 같은 오탐률:** GRU는 악성 점수가 임계값보다 클 때(`>`) 악성이다. 검증 GRU 점수 가운데 검증 오탐 수가 캐스케이드의 검증 오탐 수 이하가 되는 가장 작은 값을 임계값으로 쓴다.
- **실험 5 같은 재현율:** 검증 탐지 수가 캐스케이드의 검증 탐지 수 이상이 되는 가장 큰 값을 임계값으로 쓴다.
- **실험 5 재검사 예산과 구간:** 캐스케이드의 검증 경로는 검증 행 안에서 재검사 예산을 계산한다. 시험 차이의 95% 구간은 부트스트랩 10,000회(seed 11024)다. URL 무작위는 행(재현율은 악성 행, 오탐률은 정상 행)을, 도메인은 등록 도메인 군집을 다시 뽑는다.
- **실험 6 측정 사양:** v1 `paired_time.ready`의 측정 본문을 19개 사양으로 옮겼다. 1,000건은 v1 `batch-{split}.json`(생성 seed 46)이고, 실행 순서 seed는 47, 절약 시간 부트스트랩은 seed 509로 10,000회다.
- **실험 6 재검사 예산:** 1,000건 배치 안의 종료 후보로 예산을 다시 계산하므로, 1%에서는 실제 재검사 건수가 0일 수 있다.
- **실험 6과 v1의 비교:** v1과 다른 측정 회차이므로 0%·5%는 호출 수만 완전 일치를 단언하고, 시간은 구간 겹침과 부호를 기록한다.
- **실험 6 스레드:** v1 `inputs.py`가 TensorFlow를 가져오기 전에 TF·OMP·OPENBLAS·MKL 스레드 환경 변수를 1로 두고, 측정 함수는 threadpoolctl로 BLAS·OpenMP를 1로 제한한다.

## 실행 당시 코드와 다른 점

v2 결과와 보고서를 만든 코드와 비교하면 다음만 다르다. 계산과 문서 내용은 바뀌지 않는다.

- `v2paths.py`(새 파일)와 `v2common.py`: 폴더를 환경 변수로 지정하고 v1 폴더를 보호한다. 실행 당시에는 `final-v2/code`의 위치로 폴더를 정했다.
- `report_base.py`: 폴더를 `v2common`에서 가져온다.
- `report_v2.py`: 설명문의 실행 명령.
- `render_pages.py`: v1 보호 검사를 더하고, PDF를 연 뒤에 기존 쪽 이미지를 지운다.
- `e6_rate_timing.py`: `dryrun` 출력 폴더의 v1 보호 검사.
- Word 변환은 저장소 공용 `../upstream_partition/render_word.ps1`을 쓴다. 실행 당시 `final-v2/code/render_word.ps1`과 같은 내용이다.
- 실행 당시의 실행 메모(`RUN-NOTES.md`)는 개인 경로가 있어 올리지 않고 이 README로 대신했다.
