# Reproducing the experiments

## Source code and data

Experiment A uses the GRU, numeric branch and preprocessing from Munna Ram's `URL_CLASSIFIER` at commit `835bd5e1710531407c0b5ca09da3d7c3de8d1bbc`. The runner retrieves three Python modules and checks their SHA-256 digests. [Pinned upstream source](https://github.com/munna0912/URL_CLASSIFIER/tree/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc) · [Source and asset specification](../configs/upstream.json)

The upstream Git LFS endpoint did not serve the three CSVs. The team recovered them from a pinned earlier revision of the same repository and recovered the reference weights from the author's pinned Hugging Face revision. File sizes and SHA-256 digests matched the original LFS pointers. Experiment B trains its own weights and fits its own vocabulary. [File provenance and verification](../evidence/source-integrity.json)

## Local verification

Run from the repository root. The basic suite uses the standard library. Install NumPy to exercise the paired-bootstrap test. The upstream parity test runs once the reference modules are available.

```bash
python3.10 -m unittest discover -s tests -v
python3.10 scripts/fetch_reference.py
python3.10 -m unittest discover -s tests -p test_reference_parity.py -v
```

The last test compares the 18 numeric features and cleaned text with the pinned upstream implementation for 192 combinations of schemes, hosts and paths, including IPv4, IPv6 and Unicode characters. It requires neither training data nor TensorFlow.

## Kaggle CPU workflow

Import the appropriate notebook into Kaggle and enable internet access for setup and source retrieval. The runner installs a separate Python 3.10 environment with TensorFlow 2.12.0, NumPy 1.23.5, pandas 1.5.3 and scikit-learn 1.2.2. The environment lists record the package versions observed on each platform. [CPU environment](../environment/w1-cpu.txt) · [GPU environment](../environment/w1-gpu.txt)

1. **Prepare the data and run the pilot.** In the [preparation notebook](../notebooks/HUFS_CompSec_Cloud.ipynb), set `STAGE` successively to `diagnose`, `setup`, `fetch`, `prepare-b`, `audit-b`, `smoke` and `pilot` in the same workspace. Pass the absolute `runs/<run-id>/split` path printed by `prepare-b` as `SPLIT_DIR` for stages that need it. Smoke and pilot samples come from B's frozen training partition. [Split manifest](../evidence/b-split-manifest.json)
2. **Reproduce A.** Run the [A notebook](../notebooks/HUFS_CompSec_Experiment_A.ipynb) in a fresh CPU session. It prepares the environment and data and trains A's vocabulary and model. If a historical smoke archive is attached, the runner verifies its hash and checks restoration. The historical submitted notebook and the current English reproduction runner have distinct file identifiers. [Historical run identifiers](../evidence/experiment-a-version.json)
3. **Select the B models.** Run the [B notebook](../notebooks/HUFS_CompSec_Experiment_B.ipynb) in a separate CPU session. It can restore a verified split checkpoint or regenerate the split from the pinned sources. The manifest, training rows and model-selection rows must match their frozen hashes before training starts. [Selection code](../scripts/train_b.py)
4. **Select the gate, then evaluate the frozen policy.** Use separate private CPU sessions. The gate run compares eight thresholds on the gate-selection partition. Record the policy and archive digests before opening the internal test or external evaluation cohort. [Frozen-policy evaluation](#frozen-policy-evaluation)

Each stage writes a status file, command logs and hashes. Save them with the notebook version. Network access retrieves pinned source files and packages. Dataset URL strings are never visited.

## Experiment A internal reproduction

| Epoch 20 | Upstream saved output | Team run |
| --- | ---: | ---: |
| Validation accuracy | 98.34% | 98.40% |
| Validation binary cross-entropy | 0.0565 | 0.05769 |
| Malicious recall | Class counts unavailable | 68,895 / 70,394 (97.87%) |

The comparison uses the first training block in the upstream notebook. The team's 156,334 validation rows yielded 1,003 false positives and 1,499 false negatives. The source notebook used a TPU and did not specify the random state for this split. The team used CPU and seed 42. The approximately 0.06 percentage-point accuracy difference therefore compares different runs and potentially different validation rows. [Upstream training notebook](https://github.com/munna0912/URL_CLASSIFIER/blob/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc/Model_Training.ipynb) · [Team confusion matrix and hashes](../evidence/experiment-a-internal.json)

## Gate selection and offline prediction

In the pinned Kaggle CPU environment, restore the B training output directory `experiment-b` as `MODELS` and the frozen `split` directory as `SPLIT`. Set `MODELS_SHA256` to the verified digest of `models-frozen.json`. Use a new output directory for each run. Set `POLICY_SHA256` to the verified digest of the resulting `policy.json` before prediction.

```bash
python evaluation/gate.py --models "$MODELS" --models-sha256 "$MODELS_SHA256" \
  --split "$SPLIT" --out "$GATE_OUT"
python evaluation/predict.py --models "$MODELS" --policy "$GATE_OUT/policy.json" \
  --policy-sha256 "$POLICY_SHA256" --url 'https://example.org/demo'
```

The candidate thresholds are 0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10 and 0.20. All eight receive full-cohort extra-miss counts. Eligible candidates share the same timing sample of up to 1,000 rows, measured five times. The rule selects the smallest threshold within 5% of the fastest eligible median of run means. The runner checks actual selective execution across the full gate cohort before freezing the policy.

`gate-candidates.json` stores candidate metrics. `gate-latencies.json` stores timing sample IDs and repeated measurements. For batch CLI input, replace `--url` with `--jsonl inputs.jsonl`. Supply a unique `input_id` and a `raw_url` string on each line.

## Frozen-policy evaluation

Use each private notebook's `owner/slug` as its handle and the saved version number as its version. Generate the gate notebook from the verified B archive digest.

```bash
python3.10 evaluation/build_notebook.py --stage gate \
  --model-notebook "$B_NOTEBOOK" --model-version "$B_VERSION" \
  --archive-sha256 "$B_ZIP_SHA256" --output /tmp/HUFS_URL_Gate.ipynb
```

The gate notebook rebuilds the environment and split, evaluates the gate candidates, and saves `hufs-b-gate-policy.zip`. Download that file as `GATE_ZIP`. From the **same saved version's log**, copy the digest following `Verified gate archive` into `GATE_ZIP_SHA256` and the digest following `Frozen B policy` into `POLICY_SHA256`.

```bash
python3.10 -m tools.verify_gate_archive "$GATE_ZIP" \
  --archive-sha256 "$GATE_ZIP_SHA256" --policy-sha256 "$POLICY_SHA256"
```

The verifier checks the archive, policy and evidence digests. It recomputes additional false negatives (K) and recall loss (K/M) from the eight confusion matrices, checks T2 call rates and the five eligible timing repetitions, and reproduces the selection rule. Among 37,440 malicious gate-selection rows, at most 374 additional misses satisfy the one-percentage-point bound.

Generate the final notebook with the verified gate archive and its saved version. The runner pins the A reference archive to version 1 and its known digest.

```bash
python3.10 evaluation/build_notebook.py --stage final \
  --model-notebook "$B_NOTEBOOK" --model-version "$B_VERSION" \
  --archive-sha256 "$B_ZIP_SHA256" --reference-notebook "$A_NOTEBOOK" \
  --gate-notebook "$GATE_NOTEBOOK" --gate-version "$GATE_VERSION" \
  --gate-sha256 "$GATE_ZIP_SHA256" --output /tmp/HUFS_URL_Final.ipynb
```

The final runner verifies attached input archives or retrieves outputs from the specified notebook versions. A retry must restore the same frozen policy. The four B inference paths run on the internal test and independent external cohorts. A's external comparison uses A's weights, vocabulary and upstream external CSV preprocessing.

The `evaluation-output` directory contains policy evidence, cohort reports, 2,000 paired domain-bootstrap replicates summarized as intervals, prespecified error groups, sequential latency measurements and batch throughput. Model loading and dataset reading are timed separately. When a cohort contains no benign examples, false-positive rate and balanced accuracy are `null`. Publish aggregate metrics and digests. Keep row-level predictions and full timing logs in the private execution output.
