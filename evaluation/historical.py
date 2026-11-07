"""Compare the separate A model on the original external file after B policy freeze."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream")]
from evaluation.gate import read_json, verify_file, write_json, SPLIT_HASH
from urlcascade.adapters import vectorizer
from urlcascade.runtime import compute_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "vocab", "csv", "policy", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--policy-sha256", required=True)
    a = p.parse_args()
    compute_context("kaggle", a.out)
    verify_file(a.policy, a.policy_sha256)
    policy = read_json(a.policy)
    if (
        policy["stage"] != "gate_selection"
        or policy["experiment"] != "B"
        or policy["split_sha256"] != SPLIT_HASH
    ):
        raise ValueError("A external evaluation requires a frozen B policy")
    verify_file(
        a.model, "5f0120a7c54978ac36cb617a89c833732d772196a0c8fa8b3df7bc48f5c9e807"
    )
    verify_file(
        a.vocab, "41b33cef6e949af8cfc77336c0d77fe2d48a8d4e570b595bb77631a4d42b35c1"
    )
    verify_file(
        a.csv,
        read_json(ROOT / "configs/upstream.json")["files"][
            "dataset/df_another_validation.csv"
        ]["sha256"],
    )
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from Utils.DataProcessing import process_data

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(a.model, compile=False)
    frame = pd.read_csv(a.csv)
    ds = process_data(frame, 128, vectorizer(read_json(a.vocab)))
    cm = np.zeros((2, 2), dtype=np.int64)
    for inputs, labels in ds:
        malicious = 1 - labels.numpy().astype(int).reshape(-1)
        pred = (model(inputs, training=False).numpy().reshape(-1) < 0.5).astype(int)
        np.add.at(cm, (malicious, pred), 1)
    report = dict(
        experiment="A",
        cohort="original external CSV after upstream preprocessing",
        confusion_order=["normal", "malicious"],
        confusion=cm.tolist(),
        rows=int(cm.sum()),
        accuracy=float(np.trace(cm) / cm.sum()),
        malicious_recall=float(cm[1, 1] / cm[1].sum()),
        raw_source_rows=len(frame),
        policy_sha256=a.policy_sha256,
        upstream_external_accuracy=0.7110793343707504,
        overlap_context="W2 independent-cohort audit excludes 436693 URL overlaps and 6081 additional domain overlaps from the cleaned external pool. The historical A evaluation retains the original preprocessing.",
    )
    write_json(a.out, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
