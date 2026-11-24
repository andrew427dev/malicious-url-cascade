"""Select and calibrate all first stages using training and validation rows only."""

import argparse
import sys

from inputs import ROOT, REPO, sha, write_json, np, pd, FEATURE_NAMES

sys.path.insert(0, str(REPO / "experiments/scheme_ablation"))
from run import fit_family
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss

OMIT = (
    "abnormal_url",
    "count_dir",
    "count_embed_domian",
    "count-https",
    "count-http",
    "url_length",
    "hostname_length",
    "count-letters",
    "fd_length",
)
COLUMNS = [i for i, n in enumerate(FEATURE_NAMES) if n not in OMIT]


def learning_curve(base, x, y, tr, va):
    rng = np.random.RandomState(42)
    positions = [rng.permutation(np.flatnonzero(y[tr] == cls)) for cls in (0, 1)]
    rows = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        pos = np.sort(
            np.concatenate([ix[: int(len(ix) * fraction)] for ix in positions])
        )
        model = clone(base).fit(x[tr[pos]], y[tr[pos]])
        row = {"fraction": fraction, "n": len(pos)}
        for label, index in (("train", tr[pos]), ("validation", va)):
            p = model.predict_proba(x[index])[:, 1]
            row[label + "_ap"] = average_precision_score(y[index], p)
            row[label + "_logloss"] = log_loss(y[index], p)
        rows.append(row)
    return rows


def fit(kind):
    out = ROOT / kind
    dest = out / "stage1"
    dest.mkdir(exist_ok=False)
    parts = ("train", "validation")
    x = np.concatenate([np.load(out / (p + "-features.npy")) for p in parts])
    labels = [pd.read_csv(out / (p + "-rows.csv.gz")).y_mal.to_numpy() for p in parts]
    y = np.concatenate(labels)
    tr, va = np.split(np.arange(len(y)), [len(labels[0])])
    selection, other = {}, {}
    for family in ("logistic", "boost"):
        est, select = fit_family(family, x, y, tr, va)
        joblib.dump(est, dest / (family + ".joblib"))
        selection[family] = select
        base = est.calibrated_classifiers_[0].estimator
        select["validation_calibration"] = {}
        for key, m in [("before", base), ("sigmoid", est)]:
            pv = m.predict_proba(x[va])[:, 1]
            select["validation_calibration"][key] = {
                "brier": brier_score_loss(y[va], pv),
                "logloss": log_loss(y[va], pv),
            }
        if kind.endswith("original"):
            reduced, grid = fit_family(family, x[:, COLUMNS], y, tr, va)
            joblib.dump(reduced, dest / (family + "-omit.joblib"))
            other[family + "_omit_selection"] = grid
        if kind == "random-original":
            other[family + "_curve"] = learning_curve(base, x, y, tr, va)
    length = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    )
    length.fit(x[tr, 13:14], y[tr])
    joblib.dump(length, dest / "length.joblib")
    write_json(
        dest / "t1-selected.json",
        {
            "selection": selection,
            "supplemental": other,
            "hashes": {p.stem: sha(p) for p in dest.glob("*.joblib")},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    fit(parser.parse_args().kind)
