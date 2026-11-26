"""Experiment 4 (with supplementary A): what happens when only the http(s):// prefix is removed?

v1 scored every team test URL twice with the same frozen models: once as stored and once with
the leading http(s):// removed (paired-scores.npz). It reported only the direction of each
changed prediction. This script rebuilds both routes from the stored scores, checks the change
counts against fixed-prefix.json, and then compares each change with the true label: lost
detections and new detections of malicious URLs, removed and new false positives of benign URLs.
It also records the malicious share by prefix in the training and test rows.
"""

import numpy as np
import pandas as pd

from v2common import V1, RESULTS, MAIN, FAMILIES, load_test, taus, read_json, write_json
import evaluate as ev
from inputs import PREFIX


def main():
    rows, overall, shares = [], [], []
    for split, kind in MAIN.items():
        out = V1 / kind
        frame, s = load_test(kind)
        ids, y = s["id"], s["y"]
        with np.load(out / "paired-scores.npz") as arr:
            p = {k: arr[k] for k in arr.files}
        assert np.array_equal(p["id"], ids) and np.array_equal(p["y"], y)
        t = taus(kind)
        masks, res = ev.configurations(y, ids, s["gru"], {f: s[f] for f in FAMILIES}, t)
        # Routes for the stripped input are recomputed: the recheck budget depends on the whole batch.
        amasks, ares = ev.configurations(y, ids, p["gru"], {f: p[f] for f in FAMILIES}, t)
        stored = read_json(out / "fixed-prefix.json")
        has = frame.url.str.match(PREFIX).to_numpy()
        for name in masks:
            for field in ("tp", "fp", "fn", "tn", "t2_calls", "additional_misses"):
                assert ares[name][field] == stored["stripped"][name][field], (kind, name, field)
            a = (s["gru"] > 0.5) & masks[name]
            b = (p["gru"] > 0.5) & amasks[name]
            for where, sel in (("with_prefix", has), ("without_prefix", ~has)):
                flips = stored["pairs"][where]["flips"][name]
                assert int(((a != b) & sel).sum()) == flips["changed"]
                assert int((a & ~b & sel).sum()) == flips["malicious_to_benign"]
                assert int((~a & b & sel).sum()) == flips["benign_to_malicious"]
                mal, ben = (y == 1) & sel, (y == 0) & sel
                rows.append(dict(
                    split=split, kind=kind, name=name, group=where, rows=int(sel.sum()),
                    malicious=int(mal.sum()), benign=int(ben.sum()),
                    changed=int(((a != b) & sel).sum()),
                    lost_detection=int((a & ~b & mal).sum()),      # malicious: detected -> missed
                    new_detection=int((~a & b & mal).sum()),       # malicious: missed -> detected
                    removed_false_positive=int((a & ~b & ben).sum()),  # benign: flagged -> cleared
                    new_false_positive=int((~a & b & ben).sum()),  # benign: cleared -> flagged
                    detected_before=int((a & mal).sum()), detected_after=int((b & mal).sum()),
                    fp_before=int((a & ben).sum()), fp_after=int((b & ben).sum()),
                ))
            r0, r1 = res[name], ares[name]
            overall.append(dict(split=split, kind=kind, name=name,
                                recall_original=r0["recall"], recall_stripped=r1["recall"],
                                recall_change_pp=100 * (r1["recall"] - r0["recall"]),
                                fpr_original=r0["fpr"], fpr_stripped=r1["fpr"],
                                fpr_change_pp=100 * (r1["fpr"] - r0["fpr"]),
                                accuracy_original=r0["accuracy"], accuracy_stripped=r1["accuracy"],
                                t2_calls_original=r0["t2_calls"], t2_calls_stripped=r1["t2_calls"]))
        controls = read_json(out / "controls-final.json")["raw_scheme_by_label"]
        for part in ("train", "test"):
            for flag, label in (("True", "http(s):// 있음"), ("False", "없음")):
                c = controls[part][flag]
                shares.append(dict(split=split, kind=kind, part=part, prefix=label, n=c["n"],
                                   malicious=c["malicious"], benign=c["benign"],
                                   malicious_share=c["malicious"] / c["n"]))
        # The test counts in controls-final.json must equal the rows used above.
        tc = controls["test"]
        assert tc["True"]["n"] == int(has.sum()) and tc["True"]["malicious"] == int(y[has].sum())
        assert tc["False"]["n"] == int((~has).sum())
    pd.DataFrame(rows).to_csv(RESULTS / "e4-prefix-flips.csv", index=False)
    pd.DataFrame(overall).to_csv(RESULTS / "e4-prefix-overall.csv", index=False)
    pd.DataFrame(shares).to_csv(RESULTS / "e4-prefix-share.csv", index=False)
    write_json(RESULTS / "e4-checks.json", {
        "status": "PASS",
        "checks": ["stripped route counts equal fixed-prefix.json for 14 configurations",
                   "changed / malicious_to_benign / benign_to_malicious equal fixed-prefix.json pairs for 28 groups",
                   "test prefix counts equal controls-final.json raw_scheme_by_label"],
    })
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(overall).round(4).to_string())
    print(pd.DataFrame(shares).round(4).to_string())


if __name__ == "__main__":
    main()
