"""Experiment 5: is the cascade better than simply moving the GRU threshold?

Early exit removes some GRU false positives together with some detections. A GRU alone with a
higher decision threshold can also trade recall for fewer false positives. For each frozen
cascade this script reads the cascade's false-positive rate on the validation rows, picks the
smallest GRU threshold whose validation FPR is at most that value, and compares malicious recall
on the test rows (paired bootstrap). The secondary comparison fixes the validation recall
instead and compares test FPR. Every threshold is chosen on validation rows only.

GRU validation scores are new inference with the frozen v1 weights; they are first checked
against the validation grid stored in routing.json when the thresholds were frozen.
"""

import numpy as np
import pandas as pd

from v2common import V1, RESULTS, MAIN, FAMILIES, load_test, taus, read_json, write_json
import evaluate as ev
from experiment import TAUS, route, cascade_metrics

DRAWS = 10000
SEED = 11024


def validation_scores(kind):
    out = V1 / kind
    model = ev.load_gru(out)
    stages = ev.load_stages(out)
    frame, y, tok, x = ev.load_rows(out, "validation")
    gru = ev.score(model, tok, x)
    first = {f: stages[f].predict_proba(x)[:, 1] for f in FAMILIES}
    # Reproduce the frozen validation grid exactly (same call as evaluate.freeze).
    frozen = read_json(out / "routing.json")
    assert frozen["validation_rows"] == len(y)
    for f in FAMILIES:
        grid = frozen["families"][f]["grid"]
        assert [g["tau"] for g in grid] == TAUS
        for g in grid:
            mask, _ = route(first[f], frame.index, g["tau"], 0, "basic")
            m = cascade_metrics(y, gru, mask)
            for field in ("tp", "fp", "fn", "tn", "t2_calls", "additional_misses"):
                assert m[field] == g[field], (kind, f, g["tau"], field, m[field], g[field])
            assert abs(m["recall_loss"] - g["recall_loss"]) < 1e-12
    ids = np.asarray(frame.index, dtype=str)
    np.savez_compressed(RESULTS / f"e5-validation-scores-{kind}.npz", id=ids, y=y, gru=gru, **first)
    return frame, ids, y, gru, first


def threshold_for_fpr(gru, y, max_fp):
    """Smallest t with #(benign gru > t) <= max_fp, among validation score values."""
    benign = np.sort(gru[y == 0])[::-1]
    return float(benign[max_fp]) if max_fp < len(benign) else -np.inf


def threshold_for_recall(gru, y, min_tp):
    """Largest validation score value t with #(malicious gru > t) >= min_tp."""
    mal = np.sort(gru[y == 1])[::-1]
    need = mal[min_tp - 1]
    lower = np.unique(gru[gru < need])
    return float(lower[-1]) if len(lower) else -np.inf


def paired_interval(a, b, groups):
    """Bootstrap the difference mean(a)-mean(b) over rows (groups=None) or clusters of rows."""
    d = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(SEED)
    if groups is None:
        values, freq = np.unique(d, return_counts=True)
        draws = rng.multinomial(len(d), freq / freq.sum(), size=DRAWS)
        est = draws @ values / len(d)
    else:
        ix, _ = pd.factorize(groups, sort=True)
        pairs = np.column_stack([np.bincount(ix, weights=d).astype(int), np.bincount(ix)])
        uniq, freq = np.unique(pairs, axis=0, return_counts=True)
        draws = rng.multinomial(len(pairs), freq / freq.sum(), size=DRAWS)
        est = (draws @ uniq[:, 0]) / (draws @ uniq[:, 1])
    lo, hi = np.quantile(est, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def main():
    rows = []
    for split, kind in MAIN.items():
        vframe, vids, vy, vgru, vfirst = validation_scores(kind)
        frame, s = load_test(kind)
        ids, y, gru = s["id"], s["y"], s["gru"]
        t = taus(kind)
        _, vres = ev.configurations(vy, vids, vgru, vfirst, t)
        masks, res = ev.configurations(y, ids, gru, {f: s[f] for f in FAMILIES}, t)
        domains = frame.domain.to_numpy()
        mal, ben = y == 1, y == 0
        vben_n, vmal_n = int((vy == 0).sum()), int((vy == 1).sum())
        for name in masks:
            if name == "gru":
                continue
            casc = (gru > 0.5) & masks[name]
            # Equal FPR: GRU threshold from the cascade's validation false positives.
            t_fpr = threshold_for_fpr(vgru, vy, vres[name]["fp"])
            g_fpr = gru > t_fpr
            assert ((vgru > t_fpr) & (vy == 0)).sum() <= vres[name]["fp"]
            diff, lo, hi = paired_interval(g_fpr[mal], casc[mal], None if split == "random" else domains[mal])
            # Equal recall: GRU threshold from the cascade's validation detections.
            t_rec = threshold_for_recall(vgru, vy, vres[name]["tp"])
            g_rec = gru > t_rec
            assert ((vgru > t_rec) & (vy == 1)).sum() >= vres[name]["tp"]
            fdiff, flo, fhi = paired_interval(g_rec[ben], casc[ben], None if split == "random" else domains[ben])
            rows.append(dict(
                split=split, kind=kind, name=name,
                val_cascade_fpr=vres[name]["fpr"], val_cascade_recall=vres[name]["recall"],
                gru_threshold_equal_fpr=t_fpr,
                val_gru_fpr_at_threshold=float(((vgru > t_fpr) & (vy == 0)).sum() / vben_n),
                test_cascade_recall=res[name]["recall"], test_cascade_fpr=res[name]["fpr"],
                test_gru_recall_equal_fpr=float(g_fpr[mal].mean()), test_gru_fpr_equal_fpr=float(g_fpr[ben].mean()),
                recall_diff_gru_minus_cascade_pp=100 * diff, recall_diff_lo_pp=100 * lo, recall_diff_hi_pp=100 * hi,
                gru_threshold_equal_recall=t_rec,
                val_gru_recall_at_threshold=float(((vgru > t_rec) & (vy == 1)).sum() / vmal_n),
                test_gru_recall_equal_recall=float(g_rec[mal].mean()), test_gru_fpr_equal_recall=float(g_rec[ben].mean()),
                fpr_diff_gru_minus_cascade_pp=100 * fdiff, fpr_diff_lo_pp=100 * flo, fpr_diff_hi_pp=100 * fhi,
                test_gru_calls=len(y), test_cascade_calls=res[name]["t2_calls"],
                interval=("recall: malicious-row bootstrap / fpr: benign-row bootstrap" if split == "random" else
                          "recall: registered domains of malicious rows / fpr: registered domains of benign rows (cluster bootstrap)"),
            ))
        rows.append(dict(split=split, kind=kind, name="gru_at_0.5",
                         val_cascade_fpr=vres["gru"]["fpr"], val_cascade_recall=vres["gru"]["recall"],
                         test_cascade_recall=res["gru"]["recall"], test_cascade_fpr=res["gru"]["fpr"]))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e5-equal-fpr.csv", index=False)
    write_json(RESULTS / "e5-checks.json", {
        "status": "PASS",
        "checks": ["new GRU validation scores reproduce routing.json grid (tp/fp/fn/tn/t2_calls/additional_misses) for 2 kinds x 2 families x 12 thresholds",
                   "equal-FPR thresholds keep validation false positives <= cascade validation false positives",
                   "equal-recall thresholds keep validation detections >= cascade validation detections"],
        "rule": "malicious iff GRU malicious score > threshold; thresholds are validation score values",
        "bootstrap": {"draws": DRAWS, "seed": SEED},
    })
    pd.set_option("display.width", 300)
    print(df.round(5).to_string())


if __name__ == "__main__":
    main()
