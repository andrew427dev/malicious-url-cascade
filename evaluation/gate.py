"""Select the operating threshold on the frozen gate-selection set."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urlcascade.adapters import KerasT2, SklearnT1
from urlcascade.benchmark import sequential_benchmark
from urlcascade.core import (
    GATE_CANDIDATES,
    binary_metrics,
    cascade_invariants,
    infer,
    select_gate,
)
from urlcascade.data import sha256
from urlcascade.features import FEATURE_NAMES
from urlcascade.runtime import compute_context

SPLIT_HASH = "6440ae2948353ea42d5adb70a0e7a686d592496da94a8d24828707a0c6eef955"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_file(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}")
    return actual


def policy_code():
    paths = [Path(__file__).resolve(), *(ROOT / "urlcascade").glob("*.py")]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def load_policy(path, expected):
    verify_file(path, expected)
    policy = read_json(path)
    if (
        policy["experiment"] != "B"
        or policy["stage"] != "gate_selection"
        or policy["split_sha256"] != SPLIT_HASH
        or policy["tau"] not in GATE_CANDIDATES
        or policy["code_sha256"] != policy_code()
    ):
        raise ValueError("Unexpected frozen B policy or inference code")
    return policy


def load_models(directory, expected_manifest):
    import joblib
    import tensorflow as tf

    manifest = directory / "models-frozen.json"
    verify_file(manifest, expected_manifest)
    spec = read_json(manifest)
    required = {
        "t1.joblib",
        "t2.h5",
        "b-vocab.json",
        "t1-candidates.json",
        "b-epochs.csv",
    }
    if (
        spec["experiment"] != "B"
        or spec["split_sha256"] != SPLIT_HASH
        or spec["feature_names"] != list(FEATURE_NAMES)
        or set(spec["files"]) != required
        or spec["fit_partition"] != "train"
        or spec["selection_partition"] != "model_selection"
        or spec["scaler_fit_partition"] != "train"
        or spec["vocabulary_fit_partition"] != "train"
        or spec["t2_safe_threshold"] != 0.5
        or spec["malicious_class_t1"] != 1
    ):
        raise ValueError("Unexpected model selection manifest")
    for name, expected in spec["files"].items():
        verify_file(directory / name, expected)
    t1 = SklearnT1(joblib.load(directory / "t1.joblib"))
    t2 = KerasT2(
        tf.keras.models.load_model(directory / "t2.h5", compile=False),
        read_json(directory / "b-vocab.json"),
    )
    return t1, t2


def load_rows(path, expected_sha):
    verify_file(path, expected_sha)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len({r["input_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    if any(r["y_mal"] not in (0, 1) or r["y_mal"] != 1 - r["y_original"] for r in rows):
        raise ValueError("Unexpected label convention")
    return rows


def predictions(rows, **kwargs):
    results = []
    for start in range(0, len(rows), 128):
        block = rows[start : start + 128]
        outputs = infer(block, **kwargs)
        if [r["input_id"] for r in outputs] != [r["input_id"] for r in block] or any(
            r["status"] != "ok" for r in outputs
        ):
            raise ValueError(
                f"Invalid prediction or changed input order at offset {start}"
            )
        results.extend(outputs)
    return results


def candidate_table(rows, t1, t2):
    labels = [r["y_mal"] for r in rows]
    scores = [r["t1_score_mal"] for r in predictions(rows, mode="t1", t1=t1)]
    baseline = [r["label_mal"] for r in predictions(rows, mode="t2", t2=t2)]
    candidates = []
    for tau in GATE_CANDIDATES:
        called = [s >= tau for s in scores]
        pred = [label if route else 0 for label, route in zip(baseline, called)]
        checks = cascade_invariants(labels, baseline, pred, called)
        candidates.append(dict(tau=tau, **checks, metrics=binary_metrics(labels, pred)))
    return candidates


def select(rows, t1, t2, out, model_hash, split_file_hash):
    candidates = candidate_table(rows, t1, t2)
    timings = sequential_benchmark(
        rows, t1=t1, t2=t2, thresholds=[c["tau"] for c in candidates if c["allowed"]]
    )
    write_json(out / "gate-latencies.json", timings)
    for c in candidates:
        c["mean_latencies"] = [
            m["mean_seconds"]
            for m in timings["measurements"]
            if m["mode"] == "cascade" and m["tau"] == c["tau"]
        ]
    tau = select_gate(candidates)
    # Exercise the selective route across the whole cohort before freezing.
    actual = predictions(rows, mode="cascade", tau=tau, t1=t1, t2=t2)
    baseline = predictions(rows, mode="t2", t2=t2)
    for a, b in zip(actual, baseline):
        called = a["t1_score_mal"] >= tau
        if (
            a["t2_called"] != called
            or a["label_mal"] != (b["label_mal"] if called else 0)
            or (a["t2_score_mal"] is not None) != called
        ):
            raise ValueError("Full-cohort selective routing check failed")
    chosen = next(c for c in candidates if c["tau"] == tau)
    reference = statistics.median(
        m["mean_seconds"] for m in timings["measurements"] if m["mode"] == "t2"
    )
    write_json(out / "gate-candidates.json", candidates)
    policy = dict(
        experiment="B",
        stage="gate_selection",
        tau=tau,
        models_sha256=model_hash,
        split_sha256=SPLIT_HASH,
        gate_file_sha256=split_file_hash,
        selected_at_utc=datetime.now(timezone.utc).isoformat(),
        gate_rows=len(rows),
        additional_fn=chosen["additional_fn"],
        malicious=chosen["malicious"],
        recall_drop=chosen["recall_drop"],
        call_rate=chosen["call_rate"],
        t2_median_mean_seconds=reference,
        cascade_median_mean_seconds=statistics.median(chosen["mean_latencies"]),
        code_sha256=policy_code(),
        evidence_sha256={
            p.name: sha256(p)
            for p in (out / "gate-candidates.json", out / "gate-latencies.json")
        },
    )
    write_json(out / "policy.json", policy)
    return policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--models-sha256", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    compute_context("kaggle", args.out)
    verify_file(args.split / "manifest.json", SPLIT_HASH)
    split = read_json(args.split / "manifest.json")
    expected = split["files"]["gate_selection.jsonl"]
    rows = load_rows(args.split / "gate_selection.jsonl", expected)
    if len(rows) != split["summary"]["gate_selection"]["rows"] or any(
        r["split"] != "gate_selection" for r in rows
    ):
        raise ValueError("Expected the complete frozen gate selection set")
    t1, t2 = load_models(args.models, args.models_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    print(
        json.dumps(
            select(rows, t1, t2, args.out, args.models_sha256, expected), indent=2
        )
    )


if __name__ == "__main__":
    main()
