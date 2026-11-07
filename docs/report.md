# Reproducing an Existing Malicious URL Detection Model and Analyzing a Confidence-Based Two-Stage Cascade

Hankuk University of Foreign Studies · Computer Security · Technical report draft

## Abstract

A character-level recurrent classifier spends computation on every URL, including benign strings that a simpler model might identify reliably. We reproduced an existing character 3-gram gated recurrent unit (GRU) classifier and designed a second experiment in which logistic regression can return an early benign decision. In the reproduction, the GRU detected 68,895 of 70,394 malicious validation URLs. Its accuracy at epoch 20 was 98.40%, compared with 98.34% in the upstream notebook. For the cascade experiment, we partitioned 779,923 URLs by domain group and used a separate model-selection cohort to freeze the logistic model and the GRU. This draft reports verified results through model selection. The remaining research question is whether the frozen cascade can reduce end-to-end latency on unseen domains while keeping additional malicious misses within the prespecified limit.

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

## 7. Held-out evaluation and limits of interpretation

The final private CPU session restores the fixed A, B and gate artifacts. It compares always-benign, T1 alone, T2 alone and the cascade on the internal test and independent external cohorts. For each cohort the report records class counts, confusion matrices, malicious recall, additional misses and T2 call rate. It measures batch-1 latency and batch-128 throughput separately from model loading and data reading. The reduction in calls and the reduction in latency have separate reference values. The team targets 30% fewer T2 calls and 20% lower mean request latency relative to T2 alone. [Final evaluation implementation](../evaluation/final.py)

A paired bootstrap resamples domain groups 2,000 times so that related URLs stay together and both models are evaluated on the same draws. Prespecified failure summaries group URLs by source, length, presence of a scheme and whether the host is an IP address. Reporting the group denominator alongside additional misses prevents a large group from looking disproportionately risky solely because it contains more URLs.

The current evidence shows that A reproduced the upstream run's internal accuracy to about 0.06 percentage points, and B's model selection produced fixed artifacts before held-out scoring. The selected threshold, held-out recall, external error counts and speedup remain open until the archived gate and final reports are verified. A single split seed limits conclusions about split-to-split variation. The all-malicious external cohort limits conclusions about generalization to benign traffic. A shorter inference path may still provide little latency benefit if feature extraction and the gate consume the saved time.
