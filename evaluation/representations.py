"""Count internal feature-vector collisions after freezing the B vocabulary."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.gate import SPLIT_HASH, read_json, verify_file, write_json
from urlcascade.adapters import preprocess_link, vectorizer
from urlcascade.features import extract_features
from urlcascade.runtime import compute_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "split", "policy", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--policy-sha256", required=True)
    a = p.parse_args()
    compute_context("kaggle", a.out)
    verify_file(a.policy, a.policy_sha256)
    policy = read_json(a.policy)
    if (
        policy["experiment"] != "B"
        or policy["stage"] != "gate_selection"
        or policy["split_sha256"] != SPLIT_HASH
    ):
        raise ValueError("Representation audit requires a frozen B gate")
    verify_file(a.models / "models-frozen.json", policy["models_sha256"])
    vocab = a.models / "b-vocab.json"
    verify_file(
        vocab, read_json(a.models / "models-frozen.json")["files"]["b-vocab.json"]
    )
    verify_file(a.split / "manifest.json", SPLIT_HASH)
    target = a.split / "internal.jsonl"
    verify_file(target, read_json(a.split / "manifest.json")["files"][target.name])
    import numpy as np
    import tensorflow as tf
    from itertools import islice

    layer = vectorizer(read_json(vocab))
    counts = Counter()
    with target.open() as stream:
        while lines := list(islice(stream, 512)):
            urls = [json.loads(line)["raw_url"] for line in lines]
            tokens = (
                layer(tf.constant([preprocess_link(url) for url in urls]))
                .numpy()
                .astype("<i4")
            )
            nums = np.asarray([extract_features(url) for url in urls], dtype="<f4")
            for t, n in zip(tokens, nums):
                counts[hashlib.sha256(t.tobytes() + n.tobytes()).digest()] += 1
    report = dict(
        rows=sum(counts.values()),
        distinct_inputs=len(counts),
        collision_keys=sum(n > 1 for n in counts.values()),
        rows_in_collisions=sum(n for n in counts.values() if n > 1),
        split_sha256=SPLIT_HASH,
        vocab_sha256=read_json(a.models / "models-frozen.json")["files"][
            "b-vocab.json"
        ],
    )
    write_json(a.out, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
