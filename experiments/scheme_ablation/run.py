"""Scheme shortcut ablation on the frozen clean-rerun splits (first-stage LR/HGB only).

The GRU text branch already strips http://, https:// and www. (adapters.preprocess_link),
so scheme presence can reach the models only through the 18 numeric features
(e.g. hostname_length is 0 and abnormal_url is ~0 whenever the scheme is absent).

Conditions, all on the same frozen train / shared validation / test rows of PR #3:
  original        features of url_key as in PR #3 (reproduction check)
  scheme_removed  leading http(s):// removed before feature extraction, then refit
Selection and calibration follow experiment.py: same grids, validation AP, sigmoid on the
shared validation, threshold 0.5. The test set is never used for selection.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clean_rerun"))
from common import ROOT, SOURCE, validation_indices, write_json  # noqa: E402
from verify_inputs import verified_rows  # noqa: E402

sys.path.insert(0, str(SOURCE))
from urlcascade.features import extract_features  # noqa: E402



def metrics(y, score, pred):
    """Same definition as experiment.metrics; copied to avoid importing TensorFlow."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "pr_auc_ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "fpr": float(fp / (tn + fp)),
    }


SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def strip_scheme(url):
    return SCHEME.sub("", url, count=1)


def self_test():
    assert strip_scheme("https://a.example/x") == "a.example/x"
    assert strip_scheme("HTTP://a.example") == "a.example"
    assert strip_scheme("a.example/http://b") == "a.example/http://b"
    # hostname_length (index 14) is the scheme shortcut this ablation removes.
    assert extract_features("http://a.example/x")[14] > 0
    assert extract_features("a.example/x")[14] == 0


def features(urls):
    return np.asarray([extract_features(u) for u in urls], dtype=np.float32)


def fit_family(family, x, y, tr, va):
    entries, fitted = [], []
    for setting in [0.1, 1.0, 10.0] if family == "logistic" else [7, 15, 31]:
        if family == "logistic":
            est = make_pipeline(
                StandardScaler(), LogisticRegression(C=setting, max_iter=5000, random_state=42)
            )
        else:
            est = HistGradientBoostingClassifier(
                max_iter=60,
                max_leaf_nodes=setting,
                learning_rate=0.1,
                early_stopping=False,
                random_state=42,
            )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            est.fit(x[tr], y[tr])
        if any(issubclass(w.category, ConvergenceWarning) for w in ws):
            raise RuntimeError("convergence_failure")
        ap =float(average_precision_score(y[va], est.predict_proba(x[va])[:, 1]))
        entries.append({"setting": setting, "validation_ap": ap})
        fitted.append(est)
    best = int(np.argmax([e["validation_ap"] for e in entries]))
    calibrated = CalibratedClassifierCV(fitted[best], cv="prefit", method="sigmoid").fit(
        x[va], y[va]
    )
    return calibrated, {"entries": entries, "winner": entries[best]["setting"]}


def scheme_table(has, y, idx):
    out = {}
    for flag in (True, False):
        sel = idx[has[idx] == flag]
        out["with_scheme" if flag else "without_scheme"] = {
            "rows": int(len(sel)),
            "malicious": int(y[sel].sum()),
            "benign": int(len(sel) - y[sel].sum()),
        }
    return out


def main():
    self_test()
    data = ROOT / "data-final"
    df, file_check = verified_rows(data)
    y = df.y_mal.to_numpy()
    x_orig = np.load(data / "features.npy")
    urls = df.url_key.tolist()
    has = np.array([bool(SCHEME.match(u)) for u in urls])
    stripped = [strip_scheme(u) for u in urls]
    probe = np.random.default_rng(0).choice(len(urls), 2000, replace=False)
    if not np.array_equal(features([urls[i] for i in probe]), x_orig[probe]):
        raise RuntimeError("features_npy_mismatch")
    x_strip = features(stripped)
    results = {
        "note": __doc__,
        "python": sys.version.split()[0],
        "data_hash_equal_to_pr3": file_check,
        "splits": {},
    }
    for kind in ("random", "domain"):
        idx = np.load(data / (kind + "_indices.npz"))
        parts = {k: idx[k] for k in idx.files}
        tr, te = parts["train"], parts["test"]
        va = validation_indices(parts)
        train_keys = {stripped[i] for i in tr}
        collided = np.array([stripped[i] in train_keys for i in te])
        r = {
            "scheme_by_label": {
                "train": scheme_table(has, y, tr),
                "test": scheme_table(has, y, te),
            },
            "scheme_only_rule_test": metrics(y[te], has[te].astype(float), has[te]),
            "test_rows_whose_stripped_url_is_in_train": int(collided.sum()),
            "models": {},
        }
        for family in ("logistic", "boost"):
            m_orig, sel_orig = fit_family(family, x_orig, y, tr, va)
            m_strip, sel_strip = fit_family(family, x_strip, y, tr, va)
            p_orig = m_orig.predict_proba(x_orig[te])[:, 1]
            p_strip = m_strip.predict_proba(x_strip[te])[:, 1]
            keep = ~collided
            r["models"][family] = {
                "selection": {"original": sel_orig, "scheme_removed": sel_strip},
                "original": metrics(y[te], p_orig, p_orig >= 0.5),
                "scheme_removed": metrics(y[te], p_strip, p_strip >= 0.5),
                "scheme_removed_excluding_collisions": metrics(
                    y[te][keep], p_strip[keep], p_strip[keep] >= 0.5
                ),
            }
            print(kind, family, json.dumps(r["models"][family]["original"]["pr_auc_ap"]),
                  json.dumps(r["models"][family]["scheme_removed"]["pr_auc_ap"]), flush=True)
        results["splits"][kind] = r
    out = ROOT / "scheme-ablation"
    out.mkdir(exist_ok=False)
    write_json(out / "results.json", results)
    print("DONE", out / "results.json")


if __name__ == "__main__":
    main()
