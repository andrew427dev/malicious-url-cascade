# 공개 CSV에서 새로 학습한 악성 URL 탐지 결과

[한국어 보고서](../../docs/upstream-partition-20261124/report-ko.pdf) · [재실행 명령과 실험 조건](../../experiments/upstream_partition/README.md)

| 파일 | 읽는 방법 |
|---|---|
| [전체 시험 성능](summary/performance.csv) | 무작위·등록 도메인 분할, 원래·통일 표기의 7개 구성. 추가 미탐은 같은 조건의 GRU 단독과 비교한다. |
| [1,000건 실제 처리 시간](summary/timing.csv) | 분할 안에서 같은 1,000건을 사용했다. 단위는 ms이며 시간 단축률이 음수이면 더 오래 걸렸다. |
| [공통 행 비교](summary/common-test.json) | 원래 표기와 접두어 통일 후 재학습한 결과를 공통 원본 행으로 비교한다. |
| [원저자 내부 검증과 별도 평가](author-validation/results.json) | 첫 학습 블록의 두 평가. 세 파일 전체를 사용한 최종 모델의 학습 이력은 author-final/trained.json에 있다. |
| [분할 전 정제](team-source.json) · [전처리 표본 대조](parity.json) · [특징 예시](feature-example.json) | 공개 파일부터 원본 URL·라벨 충돌과 중복을 거친 행 수, 원저자와 대조한 입력 및 18개 수치를 확인한다. |
| [수치 검증](summary/checks.json) · [제출본 검수](document-checks.json) | 행 단위 판정과 측정 원본에서 통계량을 재계산하고 제출본의 장·쪽 번호·표·그림과 대조했다. |

각 조건 폴더에 순서가 보존된 행 ID와 라벨(*-rows.csv.gz), 예측 점수(scores.npz), 분기 기준(routing.json), 30회 무작위 재검사·민감도(sensitivity.json), 보조 기준선(controls-final.json), 1,000건의 ID·순서·구간별 실측 시간(timing.json)이 있다. 원래 표기의 fixed-prefix.json과 paired-scores.npz는 고정한 모델의 쌍대 평가다. ID는 공개 CSV 이름과 헤더를 포함한 행 번호로 구성한다. type이 0이면 악성이고 y_mal이 1이면 악성이다.

실행 당시 학습 코드 세 파일은 training-source.tar.gz에, 당시 사용한 입력·어휘·실행 코드의 해시는 조건별 training-inputs.json에 보관했다. 해당 실행의 입력 해시는 학습 중 본 시험 전에 기록했다. 원래 표기 두 조건의 도메인 메타데이터를 복구한 후 네 조건의 전체 행에 대해 독립적으로 어휘·라벨·두 입력과 분할 간 겹침을 대조했다. 현재 재실행 코드는 학습 전에 해시를 기록한다. 로컬 전체 실행 폴더에는 새 가중치, 수치·토큰 배열과 어휘도 보관했다.
