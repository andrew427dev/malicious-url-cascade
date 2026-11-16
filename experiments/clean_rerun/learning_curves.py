"""Post-hoc train-size diagnostics; never refit or select the final test models."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import ROOT, sha, write_json, validation_indices


def main():
    target = ROOT / "learning-curves.json"
    if target.exists():
        raise RuntimeError("learning_curves_already_exist")
    data = ROOT / "data-final"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in ("clean.csv", "features.npy", "random_indices.npz"):
        assert sha(data / name) == manifest["files"][name], name
    df = pd.read_csv(data / "clean.csv", usecols=["id", "y_mal"], keep_default_na=False)
    x = np.load(data / "features.npy")
    y = df.y_mal.to_numpy()
    parts = np.load(data / "random_indices.npz")
    tr = parts["train"]
    va = validation_indices(parts)
    assert not np.intersect1d(tr, va).size
    selected = json.loads((ROOT / "random/t1-selection.json").read_text(encoding="utf-8"))
    rng = np.random.RandomState(42)
    positions = {label: rng.permutation(np.flatnonzero(y[tr] == label)) for label in (0, 1)}
    result = {
        "status": "PASS",
        "purpose": "post-hoc diagnostic; final models, thresholds and test results unchanged",
        "selection_data": "shared full validation partition; no test scoring; apparent validation after model selection",
        "data_manifest_sha256": sha(data / "manifest.json"),
        "code_sha256": sha(__file__),
        "validation_rows": len(va),
        "validation_benign": int((y[va] == 0).sum()),
        "validation_malicious": int((y[va] == 1).sum()),
        "validation_ids_sha256": hashlib.sha256("\n".join(df.id.iloc[va]).encode()).hexdigest(),
        "seed": 42,
        "families": {"logistic": [], "boost": []},
    }
    stored_indices = {}
    previous = set()
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        pos = np.sort(np.concatenate([p[: int(len(p) * fraction)] for p in positions.values()]))
        ix = tr[pos]
        assert previous.issubset(set(ix))
        previous = set(ix)
        assert not np.intersect1d(ix, parts["test"]).size
        stored_indices[str(fraction)] = ix
        for family in ("logistic", "boost"):
            setting = selected["candidates"][family]["winner"]
            if family == "logistic":
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=setting, max_iter=5000, random_state=42)
                )
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=60,
                    max_leaf_nodes=setting,
                    learning_rate=0.1,
                    early_stopping=False,
                    random_state=42,
                )
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x[ix], y[ix])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError("learning_curve_convergence_failure")
            train_score = model.predict_proba(x[ix])[:, 1]
            valid_score = model.predict_proba(x[va])[:, 1]
            row = {
                "fraction": fraction,
                "training_rows": len(ix),
                "training_benign": int((y[ix] == 0).sum()),
                "training_malicious": int((y[ix] == 1).sum()),
                "training_ids_sha256": hashlib.sha256(
                    "\n".join(df.id.iloc[ix]).encode()
                ).hexdigest(),
                "setting": setting,
                "train_ap": float(average_precision_score(y[ix], train_score)),
                "validation_ap": float(average_precision_score(y[va], valid_score)),
                "train_logloss": float(log_loss(y[ix], train_score)),
                "validation_logloss": float(log_loss(y[va], valid_score)),
                "elapsed_s": time.perf_counter() - started,
                "warnings": [str(w.message) for w in caught],
            }
            if fraction == 1.0:
                expected = next(
                    r["model_validation_ap"]
                    for r in selected["candidates"][family]["entries"]
                    if r["setting"] == setting
                )
                assert abs(row["validation_ap"] - expected) < 1e-10, (
                    family,
                    row["validation_ap"],
                    expected,
                )
            result["families"][family].append(row)
            print(family, json.dumps(row), flush=True)
    np.savez_compressed(ROOT / "learning-curve-indices.npz", **stored_indices)
    result["indices_sha256"] = sha(ROOT / "learning-curve-indices.npz")
    write_json(target, result)
    print("LEARNING_CURVES PASS", flush=True)


if __name__ == "__main__":
    main()
