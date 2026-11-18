"""Full-cascade formatting probe: does removing the scheme let malicious URLs slip through?

Uses the frozen random-split artifacts from `experiment.py random` (GRU, calibrated LR/HGB,
frozen tau). Nothing is refitted. Only the test URLs are rewritten: the leading http(s)://
is removed and the 18 numeric features are recomputed. The GRU character input is
unchanged because preprocess_link already strips the scheme (checked below).
"""

import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clean_rerun"))
from experiment import cascade_metrics, malicious_scores, metrics, route  # noqa: E402
from common import ROOT, verify_training_artifacts, write_json  # noqa: E402
from urlcascade.adapters import preprocess_link  # noqa: E402
from urlcascade.features import extract_features  # noqa: E402
from run import strip_scheme, SCHEME  # noqa: E402


def main(kind="random"):
    out = ROOT / kind
    verify_training_artifacts(out)
    data = ROOT / "data-final"
    df = pd.read_csv(data / "clean.csv", keep_default_na=False)
    y = df.y_mal.to_numpy()
    x = np.load(data / "features.npy")
    encoded = np.load(data / (kind + "-encoded.npy"))
    te = np.load(data / (kind + "_indices.npz"))["test"]
    urls = df.url_key.iloc[te].tolist()
    stripped = [strip_scheme(u) for u in urls]
    if any(preprocess_link(a) != preprocess_link(b) for a, b in zip(urls, stripped)):
        raise RuntimeError("gru_text_changes_with_scheme")
    xs = np.asarray([extract_features(u) for u in stripped], dtype=np.float32)
    has = np.array([bool(SCHEME.match(u)) for u in urls])
    yt, ids = y[te], df.id.iloc[te].to_numpy()
    target = (yt == 1) & has
    with h5py.File(out / "t2.h5", "r") as handle:
        gru = tf.keras.models.load_model(handle, compile=False)

    def p2(num):
        return malicious_scores(gru.predict([encoded[te], num], batch_size=2048, verbose=0).ravel())

    p2o, p2s = p2(x[te]), p2(xs)
    frozen = json.loads((out / "policy-frozen.json").read_text(encoding="utf-8"))["families"]
    res = {
        "note": __doc__,
        "test_rows": int(len(te)),
        "malicious_with_scheme_rows": int(target.sum()),
        "gru_alone": {
            "original": metrics(yt, p2o, p2o > 0.5),
            "scheme_removed": metrics(yt, p2s, p2s > 0.5),
            "target_detected_original": int((p2o[target] > 0.5).sum()),
            "target_detected_scheme_removed": int((p2s[target] > 0.5).sum()),
        },
        "cascades": {},
    }
    for family in ("logistic", "boost"):
        model = joblib.load(out / (family + ".joblib"))
        tau = frozen[family]["tau"]
        p1o = model.predict_proba(x[te])[:, 1]
        p1s = model.predict_proba(xs)[:, 1]
        for policy, rate in (("basic", 0), ("boundary", 0.05)):
            mo, _ = route(p1o, ids, tau, rate, policy)
            ms, _ = route(p1s, ids, tau, rate, policy)
            det_o = (p2o > 0.5) & mo
            det_s = (p2s > 0.5) & ms
            res["cascades"][f"{family}_{policy}"] = {
                "tau": tau,
                "original": cascade_metrics(yt, p2o, mo),
                "scheme_removed": cascade_metrics(yt, p2s, ms),
                "target_detected_original": int(det_o[target].sum()),
                "target_detected_scheme_removed": int(det_s[target].sum()),
                "target_early_exit_original": int((~mo[target]).sum()),
                "target_early_exit_scheme_removed": int((~ms[target]).sum()),
            }
    dest = ROOT / "scheme-ablation" / f"cascade-probe-{kind}.json"
    write_json(dest, res)
    print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
