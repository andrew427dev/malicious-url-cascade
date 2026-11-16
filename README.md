# Malicious URL Cascade

## 2026-11-16 정제 데이터 재실험

현재 보고서는 세 원자료의 라벨 충돌·중복을 정제한 **717,793행**을 사용한다. 학습 70% / 검증 15% / 시험 15%를 기준으로 하고, **같은 검증 15% 전체**를 모델 선택·sigmoid 보정·분기 기준 선택에 사용한다. 로지스틱 회귀와 트리 부스팅을 대안적인 1단계 모델로 비교하며 도메인 그룹 분할도 별도로 재학습한다.

[한국어 보고서 PDF](docs/clean-rerun/report-ko.pdf) · [편집 가능한 보고서](docs/clean-rerun/report-ko.md) · [실험 코드와 실행 안내](experiments/clean_rerun/README.md) · [교수님 피드백 이행 확인](docs/clean-rerun/feedback-check.md) · [집계 증거](evidence/clean-rerun-shared-validation)

이전 5/5/5 검증 실험의 시험 결과를 본 후 검증 설계를 변경했다. 동일 시험 집합의 추가 평가이며 처음 보는 맹검 시험으로 주장하지 않는다. 정제는 명시한 입력 중복·상충 라벨의 해결이고, 남은 모든 라벨을 외부 정답으로 검증했다는 뜻은 아니다.

**아래 Experiment A/B 설명과 수치는 데이터 문제 발견 전의 보존 기록이다.** 위 정제 재실험의 결과·분할·시간 측정과 구별한다.

Can a lightweight classifier skip recurrent inference on likely benign URLs while keeping the loss in malicious recall within one percentage point? This three-person Computer Security project at Hankuk University of Foreign Studies measures both the extra malicious URLs missed and the time saved per request.

Experiment A reproduces Munna Ram's character 3-gram GRU. Experiment B trains a separate GRU on domain-separated data and places a logistic regression gate in front of it. The project adds a domain-overlap audit, selective GRU execution and an end-to-end timing benchmark. [Upstream model](https://github.com/munna0912/URL_CLASSIFIER/tree/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc) · [Technical report](docs/report.md) · [Reproduction guide](docs/reproduction.md)

## How the cascade classifies one URL

```mermaid
flowchart TD
    U["URL string"] --> F["Extract 18 numeric features"]
    F --> T1["Standardize features and score with logistic regression"]
    T1 --> G{"Malicious score below threshold?"}
    G -->|Yes| N["Return benign"]
    G -->|No| T2["Run character GRU and numeric branch"]
    U -. "Tokenize forwarded URLs" .-> T2
    F -. "Reuse unscaled features" .-> T2
    T2 --> R["Return malicious or benign"]
```

A T1 score equal to the threshold forwards the URL to T2. T2 outputs a benign score and classifies an exact score of 0.5 as benign. An additional false negative is a malicious URL caught by T2 alone and missed because the cascade exited early. All dataset URLs are processed as strings offline. [Inference code](urlcascade/core.py) · [Boundary tests](tests/test_core.py)

## Verified results

| Question | Observation | Evidence |
| --- | --- | --- |
| How closely did A reproduce internal validation accuracy? | At epoch 20 the upstream notebook recorded **98.34%** and the team obtained **98.40%**. The team model detected **68,895 of 70,394** malicious URLs (**97.87% recall**). | [A results and confusion matrix](evidence/experiment-a-internal.json) |
| Which models were frozen for B? | On 75,946 model-selection rows, T1 achieved its highest malicious average precision (**0.978709**) at **C = 10**. T2 attained its lowest validation binary cross-entropy (**0.118248**) at **epoch 5** of 20. | [Candidate scores, epoch history and hashes](evidence/experiment-b-selection.json) |
| Did the internal partitions share domains? | The four partitions contain 779,923 URLs. All six pairwise intersections of domain groups were empty. | [Split manifest](evidence/b-split-manifest.json) · [Overlap audit](evidence/hufs-w1-w3-result.json) |
| Why train on CPU? | On the same 50,000-row pilot at batch size 128, the second epoch took **17.35 s on CPU** and **74.15 s on a Tesla T4**. These are training times. | [Pilot measurements](evidence/hufs-w1-w3-result.json) |
| Did the smoke checkpoint survive restoration? | Three verification inputs produced identical decisions and a maximum absolute benign-score difference of zero after restoration. | [Restoration record](evidence/w4-run-observation.json) |

The gate-selection run froze **τ = 0.10**. On 86,371 gate-selection URLs, the cascade missed **291 additional malicious URLs out of 37,440** (0.7772 percentage points of recall) and called the GRU **61.48%** of the time. Median-of-mean single-request latency fell from **389.92 ms to 241.47 ms**. Independent archive verification reproduced the threshold from eight candidates and five timing runs per eligible candidate. A separate replay verified all ten functional requirements against the frozen artifacts and instrumented model calls. [Functional checks](evidence/experiment-b-functional-checks.json) · [Gate-selection evidence](evidence/experiment-b-gate.json) · [Evaluation procedure](docs/reproduction.md#frozen-policy-evaluation)

The frozen gate reduced single-request latency by **39.76%** on the internal test and **53.83%** on the all-malicious external cohort. The primary recall constraint failed in both. The cascade missed **1,008** additional malicious URLs internally (**1.4889 percentage points**) and **825** externally (**1.4017 points**). External malicious recall fell from **10.29%** with T2 alone to **8.88%** with the cascade. Among scheme-less malicious inputs in the internal test, 14.26% exited early and 6.09% were additional misses relative to T2. The additional-miss rate was higher without an explicit scheme internally and higher with an explicit scheme externally (6 of 81 malicious inputs). [Final aggregate evidence and digests](evidence/experiment-b-final.json) · [Subgroup counts and feature audit](docs/report.md#error-groups-and-representation-collisions)

## Experimental design

| | A · Upstream reproduction | B · Cascade evaluation |
| --- | --- | --- |
| Data | Upstream preprocessing and a stratified 80/20 split with seed 42 | Domain-disjoint training (470,281), model selection (75,946), gate selection (86,371) and internal test (147,325) |
| Model selection | Character 3-grams, vocabulary capacity 10,000, sequence length 50, embedding size 16, GRU size 16, batch size 128 and 20 epochs | Fit the scaler and vocabulary on training data. Select T1 by malicious average precision and T2 by validation binary cross-entropy. |
| Evaluation | Score the internal validation set using A's weights and vocabulary | Compare always-benign, T1 alone, T2 alone and the cascade. Time eligible gates whose malicious recall loss is at most one percentage point relative to T2. |

The independent external cohort contains **58,858 malicious URLs** after overlap exclusion. It supports an estimate of malicious recall. Its benign denominator is zero, so the false-positive rate is undefined. [Data audit](evidence/hufs-w1-w3-result.json) · [Timing harness](urlcascade/benchmark.py)

## Run the local checks

From the repository root, run the routing, score-boundary and partition checks with Python 3.10. The basic suite runs offline. The upstream parity and NumPy-dependent tests run when their dependencies are available.

```bash
python3.10 -m unittest discover -s tests -v
```

The reproduction guide covers source verification, notebook order, frozen archive validation and offline prediction. [Reproduction guide](docs/reproduction.md)
