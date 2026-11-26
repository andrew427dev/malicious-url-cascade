"""Shared read-only access to the v1 run for the v2 analyses.

Every v2 script reads models, scores and inputs from the frozen v1 run directory
and writes only below the v2 directory. Nothing here retrains or re-freezes a model.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from v2paths import IN_REPO, folders, refuse_protected  # noqa: F401  (IN_REPO stays a public name)

V1, V2, CODE = folders()
RESULTS = refuse_protected(V2 / "results")
refuse_protected(V2 / "report")
RESULTS.mkdir(parents=True, exist_ok=True)

# The v1 modules resolve their run directory from this variable; point it at v1 (read-only use).
os.environ["URLCASCADE_RUN_DIR"] = str(V1)
sys.path[:0] = [str(CODE / "experiments/upstream_partition"), str(CODE)]

# Main experiment: the team-cleaned data. Reference: the less-cleaned (upstream preprocessing) data.
MAIN = {"random": "team-random-original", "domain": "team-domain-original"}
REFERENCE = {"random": "random-original", "domain": "domain-original"}
FAMILIES = ("logistic", "boost")
POLICIES = (("basic", 0.0), ("random", 0.05), ("boundary", 0.05))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf8"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def taus(kind):
    return {f: v["tau"] for f, v in read_json(V1 / kind / "routing.json")["families"].items()}


def load_test(kind):
    """Return the test frame, stored row scores and labels, asserting their alignment."""
    import numpy as np
    import pandas as pd

    out = V1 / kind
    frame = pd.read_pickle(out / "test.pkl")
    with np.load(out / "scores.npz") as arr:
        scores = {k: arr[k] for k in arr.files}
    rows = pd.read_csv(out / "test-rows.csv.gz", index_col="id")
    assert np.array_equal(frame.index.to_numpy().astype(str), scores["id"])
    assert rows.index.equals(frame.index)
    assert np.array_equal(rows.y_mal.to_numpy(), scores["y"])
    assert np.array_equal(1 - frame.type.to_numpy(), scores["y"])
    return frame, scores
