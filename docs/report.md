# Reproducing an Existing Malicious URL Detection Model and Analyzing a Confidence-Based Two-Stage Cascade

Hankuk University of Foreign Studies · Computer Security · Technical report

## Abstract

A character-level recurrent classifier spends computation on every URL, including benign strings that a simpler model might identify reliably. We reproduced an existing character 3-gram gated recurrent unit (GRU) classifier and designed a second experiment in which logistic regression can return an early benign decision. In the reproduction, the GRU detected 68,895 of 70,394 malicious validation URLs. Its accuracy at epoch 20 was 98.40%, compared with 98.34% in the upstream notebook. For the cascade experiment, we partitioned 779,923 URLs by domain group and used a separate model-selection cohort to freeze the logistic model and the GRU. On the gate-selection cohort the frozen threshold of 0.10 reduced GRU calls by 38.52% and introduced 291 additional misses among 37,440 malicious examples. On the frozen internal and external cohorts, the cascade lowered single-request latency by 39.76% and 53.83%. Recall loss exceeded the one-percentage-point limit on both cohorts, reaching 1.4889 and 1.4017 percentage points, respectively. On the all-malicious external cohort, the GRU detected 10.29% and the cascade detected 8.88%. Among malicious URLs without an explicit scheme, the gate returned 1,923 of 13,488 internal inputs as benign (14.26%). Standalone T2 would have detected 822 of those inputs, giving an additional-miss rate of 6.09% in that subgroup. The external additional-miss rates were 1.39% without an explicit scheme and 7.41% with one (6 of 81).

## 1. Research questions and security setting

The classifier receives a URL string and returns a benign or malicious label. It uses lexical features available before any network connection. It does not resolve the hostname, retrieve the page, execute a script or inspect page content. Two identical strings therefore produce the same model input even if the content served at the destination changes.

We ask three questions. First, how closely can an independent run reproduce the upstream model's reported internal accuracy? Second, when a lightweight first stage handles apparently benign inputs, how much malicious recall and end-to-end processing time change relative to the same GRU run on every input? Third, where do additional false negatives arise, and does the pattern persist on domains excluded from development? A false benign decision is particularly important here because it prevents the heavier model from inspecting the URL. An early exit also has a cost because every request must first run the gate. Fewer GRU calls alone cannot establish a speedup.

## 2. Reproduction design and model architecture

**Experiment A** follows the upstream preprocessing, architecture and 20-epoch training schedule. It uses a stratified 80/20 row split and reports the final epoch, as in the original training block. **Experiment B** uses a separate training set, vocabulary and weights. Its four domain-disjoint partitions let us select models, select a gate and evaluate the resulting policy on different rows. A and B answer different questions. The estimate of recall lost to routing comes from a paired comparison within B. [Pinned upstream implementation](https://github.com/munna0912/URL_CLASSIFIER/tree/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc) · [A training code](../scripts/train_a.py) · [B training code](../scripts/train_b.py)

The upstream model processes two representations of each URL. The text branch lowercases the string, removes protocol markers, spaces and punctuation, and constructs a character 3-gram sequence with vocabulary capacity 10,000 and length 50. A 16-dimensional embedding feeds a 16-unit GRU with dropout and recurrent dropout of 0.2. The numeric branch projects 18 features through a 16-unit dense layer. The features include URL and host length, counts of separators, letters and digits, and address-pattern indicators. A dense layer combines the two branches, and a sigmoid outputs a **benign** score. At exactly 0.5, the prediction is benign. The source labels use 1 for benign and 0 for malicious. We invert that convention when reporting malicious-class metrics. [Feature implementation](../urlcascade/features.py) · [Upstream parity checks](../tests/test_reference_parity.py)

To preserve the reproduction, we kept the upstream feature order and preprocessing quirks. For example, two raw URLs can become the same character string after punctuation is removed. The numeric features can still distinguish some such pairs. We tested the 18 numeric values and cleaned text against the pinned source on 192 combinations of schemes, hosts and paths, including IPv4, IPv6 and Unicode inputs. This tests representation parity, which accuracy alone cannot demonstrate.

## 3. Data provenance and separation

The internal source combines the upstream labeled URL file with its top-domain file. The independent source is the upstream external validation CSV. We recovered unavailable LFS objects from pinned revisions and matched their sizes and SHA-256 digests against the source pointers. Data processing preserves raw strings for feature extraction. Grouping and duplicate checks use separate keys. [Source verification](../evidence/source-integrity.json) · [Data audit](../evidence/hufs-w1-w3-result.json)

For B, 784,293 internal source rows yielded 779,923 retained rows after 4,370 exclusions. Recorded reasons include unusable hosts, conflicting labels and unsupported schemes. Using a fixed public suffix list and seed 42, we assigned domain groups to ten folds and combined them into four partitions. Six pairwise checks found no shared domain group between any two internal partitions. A model fitted on one domain can otherwise appear to generalize by recognizing related URLs from that same domain in validation or test data. [Frozen split manifest](../evidence/b-split-manifest.json)

| Partition and purpose | URLs | Domain groups | Malicious | Benign |
| --- | ---: | ---: | ---: | ---: |
| Training · fit model parameters, scaler and vocabulary | 470,281 | 146,577 | 211,132 | 259,149 |
| Model selection · select logistic regularization and GRU epoch | 75,946 | 24,373 | 36,318 | 39,628 |
| Gate selection · choose the routing threshold | 86,371 | 24,313 | 37,440 | 48,931 |
| Internal test · evaluate the frozen policy | 147,325 | 48,776 | 67,699 | 79,626 |

The independent external cohort retains 58,858 malicious URLs across 29,903 groups after internal-overlap exclusion. Its composition was established by the data audit. It provides a denominator for malicious recall and additional misses. Both the false-positive rate and balanced accuracy are undefined because the benign denominator is zero. We will present A's upstream-style external comparison separately, since its preprocessing, cohort and weights differ from B's.

## 4. Execution environment and reproduction result

We used Kaggle, a dedicated Python 3.10 environment, TensorFlow 2.12.0, NumPy 1.23.5, pandas 1.5.3 and scikit-learn 1.2.2. On a fixed 50,000-row training pilot at batch size 128, epoch two took 17.35 seconds on CPU and 74.15 seconds on a Tesla T4. We therefore selected CPU for the full runs. The pilot measured training time on this workload. Inference latency requires separate measurements with the models already loaded. [Pilot and environment](../evidence/hufs-w1-w3-result.json)

Experiment A used seed 42. The source split included 156,859 validation rows before preprocessing. The evaluated cohort contained 156,334. The following matrix uses malicious as the positive class. [Verified A results](../evidence/experiment-a-internal.json)

| Actual class | Predicted benign | Predicted malicious | Total |
| --- | ---: | ---: | ---: |
| Benign | 84,937 | 1,003 | 85,940 |
| Malicious | 1,499 | 68,895 | 70,394 |
| Total | 86,436 | 69,898 | 156,334 |

Accuracy was (84,937 + 68,895) / 156,334 = **98.40%**. Malicious recall was 68,895 / 70,394 = **97.87%**, and the benign false-positive rate was 1,003 / 85,940 = **1.17%**. The first upstream training block recorded 98.34% validation accuracy and 0.0565 binary cross-entropy (BCE) at epoch 20. The corresponding team values were 98.40% and 0.05769. The upstream split did not specify the same random state, and the upstream run used a TPU. The 0.06 percentage-point difference indicates similar reported internal accuracy. Attributing it to a better model would require an identical cohort and controlled comparison. [Upstream training output](https://github.com/munna0912/URL_CLASSIFIER/blob/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc/Model_Training.ipynb)

## 5. Frozen B model selection

T1 uses the same 18 numeric features, standardized with statistics fitted on B training rows. We trained L2-regularized logistic regression and selected the inverse regularization strength, C, by average precision (AP) for the malicious class. All three candidates converged. [Selection log and model hashes](../evidence/experiment-b-selection.json)

| C | Malicious AP on the model-selection set | Solver iterations |
| ---: | ---: | ---: |
| 0.1 | 0.978556 | 58 |
| 1 | 0.978687 | 61 |
| **10** | **0.978709** | **68** |

The AP advantage of C = 10 over C = 1 was about 0.000023. This determined the choice under the recorded selection rule. AP measures ranking across thresholds. It does not establish that a particular early-exit threshold meets the recall bound.

T2 used a training-only vocabulary and the upstream GRU architecture. We examined 20 checkpoints using model-selection BCE. The minimum was **0.118248 at epoch 5**. By epoch 20, training BCE had fallen to 0.026433 while model-selection BCE was 0.138743. We retained epoch 5 and froze the model files, vocabulary, scaler pipeline and selection records before gate selection. Both the T2-only baseline and the cascade load these same T2 weights.

## 6. Gate selection and timing protocol

Let p be T1's malicious score and τ the routing threshold. If p < τ, return benign immediately. Otherwise, tokenize the URL and call T2 using the unscaled numeric features already computed. Equality forwards the request. At τ = 0 every valid row reaches T2, supplying a consistency check against T2 alone. The implementation preserves input order and duplicate strings, flags invalid inputs, and leaves the T2 score empty on early exits. [Routing implementation](../urlcascade/core.py) · [Gate selection code](../evaluation/gate.py)

For each of eight thresholds (0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10 and 0.20), count **K**, the malicious rows that T2 alone detected but the gate discharged as benign. For **M** malicious rows, the decrease in recall is **K/M** because all forwarded rows use the same T2 decision. Malicious inputs missed by both paths are excluded from K. With M = 37,440, the one-percentage-point limit allows at most 374 additional misses. A count of 375 exceeds the limit. The gate cohort establishes eligibility. The held-out cohorts measure how well it generalizes.

Eligible candidates share a deterministic 1,000-row timing sample. After warm-up, the runner processes one URL at a time, including feature extraction, routing, the model call when required and restoration of the output. It changes path order between the five repetitions. For each candidate it takes the median of the five mean request latencies and selects the smallest threshold within 5% of the fastest eligible candidate. It then reruns selective inference on the whole gate cohort and records the policy and evidence digests. [Timing harness](../urlcascade/benchmark.py) · [Independent archive verifier](../tools/verify_gate_archive.py)

The completed CPU run froze τ = 0.10. It forwarded 53,103 of 86,371 requests to T2, and its 291 additional misses reduced malicious recall by 0.7772 percentage points. The maximum allowed on this cohort was 374 misses. Median-of-mean single-request latency fell from 389.92 ms with T2 alone to 241.47 ms with the cascade, a 38.07% reduction. Independent verification reproduced all eight additional-miss counts, checked the five timing runs for each of the seven eligible thresholds, and matched the frozen policy and archive digests to the saved log. The 0.20 gate missed 610 additional malicious URLs and exceeded the 1 percentage point limit. [Gate-selection evidence](../evidence/experiment-b-gate.json)

| Threshold | Additional misses | Recall loss (pp) | T2 call rate | Median run mean (ms/URL) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.0000 | 100.00% | 388.10 |
| 0.001 | 3 | 0.0080 | 97.68% | 384.49 |
| 0.005 | 6 | 0.0160 | 94.14% | 368.65 |
| 0.01 | 9 | 0.0240 | 91.20% | 351.47 |
| 0.02 | 31 | 0.0828 | 87.11% | 335.03 |
| 0.05 | 111 | 0.2965 | 70.26% | 281.36 |
| 0.10 | 291 | 0.7772 | 61.48% | 241.47 |
| 0.20 | 610 | 1.6293 | 51.69% | Ineligible |

## 7. Held-out evaluation and limits of interpretation

The final private CPU session restores the fixed A, B and gate artifacts. It compares always-benign, T1 alone, T2 alone and the cascade on the internal test and independent external cohorts. For each cohort the report records class counts, confusion matrices, malicious recall, additional misses and T2 call rate. It measures batch-1 latency and batch-128 throughput separately from model loading and data reading. The reduction in calls and the reduction in latency have separate reference values. The team targets 30% fewer T2 calls and 20% lower mean request latency relative to T2 alone. [Final evaluation implementation](../evaluation/final.py)

A paired bootstrap resamples domain groups 2,000 times so that related URLs stay together and both models are evaluated on the same draws. Prespecified failure summaries group URLs by source, length, presence of a scheme and whether the host is an IP address. Reporting the group denominator alongside additional misses prevents a large group from looking disproportionately risky solely because it contains more URLs.

The fixed gate met both cost targets and exceeded the one-percentage-point recall-loss limit on both cohorts. The internal 95% interval for recall loss crosses 1 pp. The external interval lies entirely above the bound.

The gap between the gate-selection cohort (0.7772 pp) and the held-out cohorts (1.4889 and 1.4017 pp) is the central finding of the evaluation. The width of the internal interval, 0.6149 to 2.9864 pp, shows that the recall cost of a fixed threshold varies substantially with which domain groups a cohort contains. A threshold chosen from a single cohort's point estimate therefore carries a transfer risk that the selection rule did not account for, and the eligibility margin on the gate cohort (291 misses against a ceiling of 374) was too small to absorb it.

### Internal test

All four paths scored 147,325 URLs from 48,776 domains, including 67,699 malicious and 79,626 benign examples. The malicious class is positive.

| Path | TN | FP | FN | TP | Malicious recall | FPR | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always benign | 79,626 | 0 | 67,699 | 0 | 0.00% | 0.00% | 50.00% |
| T1 alone | 76,907 | 2,719 | 9,199 | 58,500 | 86.41% | 3.41% | 91.50% |
| T2 alone | 75,612 | 4,014 | 3,438 | 64,261 | 94.92% | 5.04% | 94.94% |
| Cascade | 76,584 | 3,042 | 4,446 | 63,253 | 93.43% | 3.82% | 94.81% |

The cascade incurred 1,008 additional misses out of 67,699 malicious URLs (1.4889 pp). The maximum allowed was 676. It forwarded 88,584 of 147,325 inputs (60.13%). The 95% paired domain-bootstrap interval for recall loss was 0.6149 to 2.9864 percentage points across 2,000 valid replicates.

Early exits also removed false positives. The cascade's benign false-positive rate was 3.82% against 5.04% for T2 alone, and precision rose from 0.9412 to 0.9541, because 58,741 inputs that T1 scored as benign never reached the GRU and could not be flagged by it. This benefit is a side effect of the same routing that produced the additional misses and does not offset them under the primary criterion.

### Independent external evaluation

All four paths scored 58,858 URLs from 29,903 domains, including 58,858 malicious and 0 benign examples. The malicious class is positive.

| Path | TN | FP | FN | TP | Malicious recall | FPR | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always benign | 0 | 0 | 58,858 | 0 | 0.00% | N/A | N/A |
| T1 alone | 0 | 0 | 54,244 | 4,614 | 7.84% | N/A | N/A |
| T2 alone | 0 | 0 | 52,804 | 6,054 | 10.29% | N/A | N/A |
| Cascade | 0 | 0 | 53,629 | 5,229 | 8.88% | N/A | N/A |

The cascade incurred 825 additional misses out of 58,858 malicious URLs (1.4017 pp). The maximum allowed was 588. It forwarded 26,130 of 58,858 inputs (44.39%). The 95% paired domain-bootstrap interval for recall loss was 1.2671 to 1.5387 percentage points across 2,000 valid replicates.

The independent external cohort contains only malicious URLs. Its false-positive rate and balanced accuracy are undefined.

All 32,728 external early exits were malicious inputs returned as benign. Standalone T2 would have detected 825 of them. The measured 53.83% latency reduction and 55.61% call reduction describe this all-malicious sample. Estimating savings on benign traffic requires a cohort that includes benign inputs.

### Measured inference cost

The batch-1 statistic is the median of five run means over 1,000 fixed inputs per cohort. Batch-128 throughput is the median of five runs over 10,000 fixed inputs. Feature extraction and selective routing are inside the timed path. File reading and model loading are measured separately.

| Cohort | Path | Batch-1 latency (ms/URL) | Batch-1 p95 (ms) | Batch-128 throughput (URLs/s) |
| --- | --- | ---: | ---: | ---: |
| Test | Always benign | 0.042 | 0.062 | 26,096.0 |
| Test | T1 alone | 0.339 | 0.396 | 10,446.3 |
| Test | T2 alone | 398.039 | 439.747 | 289.9 |
| Test | Cascade | 239.789 | 426.556 | 293.8 |
| External | Always benign | 0.040 | 0.061 | 26,801.8 |
| External | T1 alone | 0.337 | 0.400 | 14,144.1 |
| External | T2 alone | 393.429 | 433.215 | 288.7 |
| External | Cascade | 181.644 | 412.386 | 295.9 |

| Cohort | Recall loss ≤ 1 pp | GRU calls reduced ≥ 30% | Batch-1 latency reduced ≥ 20% |
| --- | --- | --- | --- |
| Test | Missed (1.4889 pp) | Met (39.87%) | Met (39.76%) |
| External | Missed (1.4017 pp) | Met (55.61%) | Met (53.83%) |

Batch-128 throughput rose from 289.9 to 293.8 URLs/s internally and from 288.7 to 295.9 externally. Batch-1 p95 latency fell from 439.7 to 426.6 ms internally and from 433.2 to 412.4 ms externally. Both changes were smaller in relative terms than the mean single-request latency reductions. We timed the full inference path. The measurements do not isolate framework overhead from model computation.

### Error groups and representation collisions

A malicious early exit is any malicious input the gate returns as benign. An additional miss is a malicious early exit that standalone T2 correctly detected. Both percentages below use all malicious inputs in the same subgroup as their denominator. The evaluator marks a scheme as present when the raw string starts with a scheme name followed by `://`.

| Cohort | Explicit scheme | Malicious inputs | Malicious early exits | Additional misses vs. T2 |
| --- | --- | ---: | ---: | ---: |
| Internal | Absent | 13,488 | 1,923 (14.26%) | 822 (6.09%) |
| Internal | Present | 54,211 | 315 (0.58%) | 186 (0.34%) |
| External | Absent | 58,777 | 32,719 (55.67%) | 819 (1.39%) |
| External | Present | 81 | 9 (11.11%) | 6 (7.41%) |

The remaining prespecified groups use the same additional-miss definition.

| Group | Internal additional misses / malicious | Independent external additional misses / malicious |
| --- | ---: | ---: |
| ip\_host=False | 1,004 / 50,113 | 817 / 55,216 |
| ip\_host=True | 4 / 17,586 | 8 / 3,642 |
| length\_Q1 | 273 / 16,017 | 255 / 23,949 |
| length\_Q2 | 380 / 22,198 | 251 / 9,300 |
| length\_Q3 | 145 / 14,874 | 232 / 13,535 |
| length\_Q4 | 210 / 14,610 | 87 / 12,074 |
| source=all\_top\_domains\_merged.csv | 0 / 0 | No rows |
| source=df\_another\_validation.csv | No rows | 825 / 58,858 |
| source=df\_final.csv | 1,008 / 67,699 | No rows |

Internally, the additional-miss rate was higher without an explicit scheme (6.09% versus 0.34%). The external rates went in the opposite direction (1.39% versus 7.41%). Although 819 of the 825 external additional misses had no explicit scheme, that subgroup contained 58,777 of the 58,858 malicious inputs. Only 81 external inputs had an explicit scheme, so six misses have a large effect on its rate.

The extractor counts `http` and `https` anywhere in the raw string. We recomputed `hostname_length`, `abnormal_url`, `count-http` and `count-https` on all inputs without an explicit scheme, including benign rows. At least one of the four features was nonzero in 112 of 86,302 internal rows and 483 of 58,777 external rows. These counts describe the inputs. Establishing which features caused a routing decision would require a separate experiment. [Feature implementation](../urlcascade/features.py)

The frozen GRU token and numeric representation produced 749,560 distinct inputs from 779,923 internal URLs. In total, 9,322 repeated input patterns covered 39,685 URLs. This counts collisions in the input representation. Establishing their effect on misclassification requires label-level analysis.

### A under original external preprocessing

With A's own final-epoch weights and vocabulary, the upstream pipeline retained 499,317 of 507,195 original external records. It produced TN = 346,999, FP = 45,695, FN = 99,285 and TP = 7,338. Malicious recall was 6.88%, and accuracy was 70.96% versus the upstream notebook's 71.11%. An independent offline replay reproduced all four confusion counts. This comparison retains the original preprocessing and has a different cohort and model from B.

### Interpretation

The single split seed and the recorded CPU environment limit transfer to new splits and hardware. In particular, B's external cohort measures malicious sensitivity after overlap exclusion. It cannot establish the benign false-positive rate or the routing cost of a representative mixed production stream. The group intervals account for within-domain dependence but do not model future changes in domain ownership or traffic mix.

The course report includes a de-identified additional miss from each cohort and the scores from its standalone and routed passes. A sweep of the other seven thresholds on held-out predictions remains a possible exploratory analysis. The reported policy was frozen before evaluation.

[Aggregate evidence and digests](../evidence/experiment-b-final.json) · [Experiment tracker](https://github.com/andrew427dev/malicious-url-cascade/issues/1)
