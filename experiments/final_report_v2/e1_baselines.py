"""Experiment 1: trivial and single-model baselines on the same held-out rows.

Professor's 11/13 reply asks for a trivial baseline (URL length) to detect shortcut
features. v1 fitted the URL-length model on training rows only but never reported it.
This script scores it, the HTTP(S)-prefix rule, each first stage alone and the GRU on the
full test set of every condition, and checks each value against the frozen v1 records.
"""

import numpy as np
import pandas as pd
import joblib

from v2common import V1, RESULTS, MAIN, REFERENCE, FAMILIES, load_test, read_json, write_json, sha
import evaluate as ev
from inputs import PREFIX


def same(a, b):
    keys = ("tn", "fp", "fn", "tp")
    return all(int(a[k]) == int(b[k]) for k in keys) and abs(a["pr_auc_ap"] - b["pr_auc_ap"]) < 1e-12


def main():
    rows, checks = [], []
    main28 = pd.read_csv(V1 / "results/main-28.csv")
    for split in ("random", "domain"):
        for arm, kind in (("team", MAIN[split]), ("reference", REFERENCE[split])):
            out = V1 / kind
            frame, s = load_test(kind)
            y = s["y"]
            x = np.load(out / "test-features.npy")
            assert len(x) == len(y)
            controls = read_json(out / "controls-final.json")
            official = read_json(out / "results.json")
            hashes = read_json(out / "stage1/t1-selected.json")["hashes"]
            assert sha(out / "stage1/length.joblib") == hashes["length"]
            length = joblib.load(out / "stage1/length.joblib")

            def add(model, score, pred, note):
                m = ev.metrics(y, score, pred)
                rows.append(dict(split=split, arm=arm, kind=kind, model=model, rule=note, **m))
                return m

            p = length.predict_proba(x[:, 13:14])[:, 1]
            m = add("url_length", p, p >= 0.5, "LR on url_length only, P>=0.5")
            checks.append(dict(kind=kind, item="url_length", equal_to_v1_controls=same(m, controls["length"])))
            has = frame.url.str.match(PREFIX).to_numpy()
            m = add("prefix_rule", has.astype(float), has, "malicious iff raw URL starts with http(s)://")
            checks.append(dict(kind=kind, item="prefix_rule", equal_to_v1_controls=same(m, controls["scheme_only"])))
            for fam in FAMILIES:
                m = add(fam + "_alone", s[fam], s[fam] >= 0.5, "calibrated first stage alone, P>=0.5")
                checks.append(dict(kind=kind, item=fam + "_alone", equal_to_v1_results=same(m, official["first_stage"][fam])))
            m = add("gru", s["gru"], s["gru"] > 0.5, "GRU alone, malicious score>0.5")
            ref = main28[(main28.kind == kind) & (main28.name == "gru")].iloc[0].to_dict()
            checks.append(dict(kind=kind, item="gru", equal_to_v1_main28=same(m, ref)))
            # Descriptive: median URL-length feature by class (feature used by the length model).
            for cls, label in ((1, "malicious"), (0, "benign")):
                rows[-1]["median_url_length_" + label] = float(np.median(x[y == cls, 13]))
            coef = length[-1].coef_.ravel()[0]
            rows[-1]["length_model_standardized_coef"] = float(coef)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e1-baselines.csv", index=False)
    ok = all(all(v for k, v in c.items() if k.startswith("equal")) for c in checks)
    write_json(RESULTS / "e1-checks.json", {"status": "PASS" if ok else "FAIL", "checks": checks})
    print(df[["split", "arm", "model", "accuracy", "precision", "recall", "f1", "pr_auc_ap", "roc_auc", "fpr"]].round(4).to_string())
    print("checks", "PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()
