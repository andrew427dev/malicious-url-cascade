"""Experiment 6: measured processing time of the recheck rates 1, 5, 10 and 20%.

v1 measured the frozen threshold with 0% and 5% recheck only. This script times, for the
team-cleaned data of each split, 19 specifications on the same 1,000 raw test URLs v1 used
(batch-{split}.json): GRU alone and, for LR and HGB, the basic exit plus random and boundary
recheck at 1/5/10/20% of the early-exit candidates in the batch. The measured path, warm-up,
30 randomized repeats, single thread and assertions follow v1 paired_time.ready; that function
only accepts its own specification list, so its body is reproduced here with the new list.
Detection on the full test rows and the recall-loss interval use v1 route/cascade_metrics and
aggregate.loss_interval with the v1 decision rule.

Usage: python e6_rate_timing.py measure   (timing only; run with nothing else heavy)
       python e6_rate_timing.py summarize
"""

import platform
import sys
from datetime import datetime
from time import perf_counter_ns

from threadpoolctl import threadpool_limits, threadpool_info

from v2common import V1, RESULTS, MAIN, FAMILIES, load_test, read_json, write_json, taus
from inputs import tf, np, pd, vectorizer, preprocess_link, extract_features
from evaluate import load_gru, load_stages, load_rows, route
from experiment import cascade_metrics
from cohorts import model_url

RATES = (0.01, 0.05, 0.10, 0.20)
REPEATS = 30


def specifications(t):
    specs = [("gru", "basic", 0.0, 0.0)]
    for fam in FAMILIES:
        specs.append((fam, "basic", 0.0, t[fam]))
        for policy in ("random", "boundary"):
            specs.extend((fam, policy, r, t[fam]) for r in RATES)
    return specs


def ready(kind, ids):
    out = V1 / kind
    frame, y, z, x = load_rows(out, "test")
    loc = frame.index.get_indexer(ids)
    assert np.all(loc >= 0)
    raw = frame.url.iloc[loc].tolist()
    x, z = x[loc], z[loc]
    with np.load(out / "scores.npz", allow_pickle=False) as saved:
        assert np.array_equal(ids, saved["id"][loc]) and np.array_equal(y[loc], saved["y"][loc])
        expected = {k: saved[k][loc] for k in ("gru", *FAMILIES)}
    layer = vectorizer(read_json(out / "vocab.json"))
    model = load_gru(out)
    gates = load_stages(out)
    t = taus(kind)
    specs = specifications(t)
    expected_mask = {s: np.ones(len(ids), bool) if s[0] == "gru" else route(expected[s[0]], ids, s[3], s[2], s[1])[0]
                     for s in specs}
    expected_pred = {s: (expected["gru"] > .5) & v for s, v in expected_mask.items()}

    @tf.function(input_signature=[tf.TensorSpec([None, 50], tf.int32), tf.TensorSpec([None, 18], tf.float32)])
    def forward(tok, num):
        return model([tok, num], training=False)

    def execute(spec):
        fam, policy, rate, tau = spec
        a = perf_counter_ns()
        urls = [model_url(u, "original", True) for u in raw]
        numeric = np.asarray([extract_features(u) for u in urls], np.float32)
        b = perf_counter_ns()
        scores = None if fam == "gru" else gates[fam].predict_proba(numeric)[:, 1]
        c = perf_counter_ns()
        mask = np.ones(len(raw), bool) if fam == "gru" else route(scores, ids, tau, rate, policy)[0]
        idx = np.flatnonzero(mask)
        pred = np.zeros(len(raw), bool)
        d = perf_counter_ns()
        tok = tf.cast(layer(tf.constant([preprocess_link(urls[i]) for i in idx])), tf.int32) if len(idx) else None
        e = perf_counter_ns()
        if len(idx):
            pred[idx] = forward(tok, numeric[idx]).numpy().ravel() < .5
        f = perf_counter_ns()
        np.testing.assert_array_equal(x, numeric)
        if tok is not None:
            np.testing.assert_array_equal(z[idx], tok.numpy())
        np.testing.assert_array_equal(expected_mask[spec], mask)
        np.testing.assert_array_equal(expected_pred[spec], pred)
        return dict(kind=kind, family=fam, policy=policy, rate=rate, tau=tau, calls=len(idx),
                    input_ms=(b - a + e - d) / 1e6, score_ms=(c - b) / 1e6, route_ms=(d - c) / 1e6,
                    gru_ms=(f - e) / 1e6, total_ms=(f - a) / 1e6)
    return specs, execute


def measure(repeats=REPEATS, dest=RESULTS):
    """repeats/dest differ only for the dry run, which never writes into results/."""
    meta = dict(start=datetime.now().astimezone().isoformat(), python=sys.version, platform=platform.platform(),
                processor=platform.processor(), tensorflow=tf.__version__, repeats=repeats, warmup=1)
    with threadpool_limits(limits=1):
        meta["threadpools"] = threadpool_info()
        meta["tf_threads"] = [tf.config.threading.get_intra_op_parallelism_threads(),
                              tf.config.threading.get_inter_op_parallelism_threads()]
        for split, kind in MAIN.items():
            batch = read_json(V1 / f"batch-{split}.json")
            ids = np.asarray(batch["ids"])
            assert len(ids) == 1000
            specs, execute = ready(kind, ids)
            for spec in specs:          # one warm-up call per specification
                execute(spec)
            rng = np.random.default_rng(47)
            rows = []
            for rep in range(1, repeats + 1):
                for order, i in enumerate(rng.permutation(len(specs)), 1):
                    rows.append(dict(repeat=rep, order=order, **execute(specs[i])))
                print(split, rep, flush=True)
            df = pd.DataFrame(rows)
            assert len(df) == repeats * len(specs) == repeats * 19
            df.to_csv(dest / f"e6-timing-{split}.csv", index=False)
            print(df.groupby(["family", "policy", "rate"]).agg(calls=("calls", "first"), total=("total_ms", "median")).to_string(), flush=True)
    meta["end"] = datetime.now().astimezone().isoformat()
    write_json(dest / "e6-timing-meta.json", meta)


def verdict(r):
    if r["family"] == "gru":
        return "기준"
    if pd.isna(r["loss_hi"]):
        return "유보"
    if r["saved_lo"] <= 0 <= r["saved_hi"] or r["loss_lo"] <= .01 <= r["loss_hi"]:
        return "유보"
    if r["saved_lo"] > 0 and r["loss_hi"] < .01:
        return "충족·단축"
    return "시간 또는 미탐 기준 미충족"


def summarize(src=RESULTS, dest=RESULTS, repeats=REPEATS):
    from aggregate import loss_interval

    v1_time = pd.read_csv(V1 / "results/timing-summary.csv")
    v1_sens = pd.read_csv(V1 / "results/sensitivity.csv")
    out, compare = [], []
    for split, kind in MAIN.items():
        d = pd.read_csv(src / f"e6-timing-{split}.csv")
        assert set(d.repeat) == set(range(1, repeats + 1))
        frame, s = load_test(kind)
        ids, y, gru = s["id"], s["y"], s["gru"]
        domains = frame.domain.to_numpy()
        t = taus(kind)
        rng = np.random.default_rng(509)
        draws = rng.integers(0, repeats, (10000, repeats))
        ref = d[d.family == "gru"].sort_values("repeat")
        for (fam, policy, rate), g in d.groupby(["family", "policy", "rate"], sort=False):
            g = g.sort_values("repeat")
            assert np.array_equal(g.repeat, ref.repeat) and g.calls.nunique() == 1
            diff = ref.total_ms.to_numpy() - g.total_ms.to_numpy()
            lo, hi = np.quantile(np.median(diff[draws], axis=1), [.025, .975])
            row = dict(split=split, kind=kind, family=fam, policy=policy, rate=rate, tau=float(g.tau.iloc[0]),
                       batch_calls=int(g.calls.iloc[0]), saved_ms=float(np.median(diff)), saved_lo=float(lo),
                       saved_hi=float(hi), saved_pct=float(np.median(100 * diff / ref.total_ms.to_numpy())),
                       min_ms=g.total_ms.min(), max_ms=g.total_ms.max(),
                       **{k: float(g[k].median()) for k in ("total_ms", "input_ms", "score_ms", "route_ms", "gru_ms")})
            # Detection on the full test rows of the same specification.
            if fam == "gru":
                mask = np.ones(len(y), bool)
                m = cascade_metrics(y, gru, mask)
                lo_l, hi_l = 0.0, 0.0
            else:
                p = s[fam]
                mask, k = route(p, ids, t[fam], rate, policy)
                m = cascade_metrics(y, gru, mask, p >= t[fam])
                m["rechecks"] = k
                lo_l, hi_l = loss_interval(y, (gru > .5) & ~mask, domains, split, False)
                ref_s = v1_sens[(v1_sens.kind == kind) & (v1_sens.family == fam) & (v1_sens.tau == t[fam])
                                & np.isclose(v1_sens.rate, rate) & (v1_sens.policy == policy)]
                assert len(ref_s) == 1
                for field in ("tp", "fp", "fn", "tn", "t2_calls", "additional_misses", "rechecks",
                              "recovered_malicious", "new_false_positives"):
                    assert int(ref_s.iloc[0][field]) == int(m[field]), (kind, fam, policy, rate, field)
            row.update(m)
            row.update(loss_lo=lo_l, loss_hi=hi_l)
            row["decision"] = verdict(row)
            out.append(row)
            # 0% and 5% also exist in v1 (measured in a different session together with three other
            # conditions): calls must match exactly; times are compared, not asserted.
            if fam == "gru" or rate in (0.0, 0.05):
                name = "gru" if fam == "gru" else fam + "_" + policy
                v = v1_time[(v1_time.kind == kind) & (v1_time.name == name) & (v1_time.main)]
                assert len(v) == 1 and int(v.iloc[0].calls) == row["batch_calls"]
                v = v.iloc[0]
                compare.append(dict(split=split, name=name, rate=rate, calls=row["batch_calls"],
                                    v1_total_ms=v.total_ms, v2_total_ms=row["total_ms"],
                                    v1_saved_ms=v.saved_ms, v1_saved_lo=v.saved_lo, v1_saved_hi=v.saved_hi,
                                    v2_saved_ms=row["saved_ms"], v2_saved_lo=row["saved_lo"], v2_saved_hi=row["saved_hi"],
                                    intervals_overlap=bool(max(v.saved_lo, row["saved_lo"]) <= min(v.saved_hi, row["saved_hi"])),
                                    same_sign=bool(np.sign(v.saved_ms) == np.sign(row["saved_ms"]))))
    df = pd.DataFrame(out)
    df.to_csv(dest / "e6-rate-summary.csv", index=False)
    pd.DataFrame(compare).to_csv(dest / "e6-v1-comparison.csv", index=False)
    write_json(dest / "e6-checks.json", {
        "status": "PASS",
        "checks": ["30 repeats x 19 specifications per split, each call asserted against stored v1 scores",
                   "full-test detection of every rate/policy equals v1 sensitivity.csv",
                   "batch GRU calls of 0% and 5% equal v1 timing-summary.csv"],
        "v1_time_overlap": f"{sum(c['intervals_overlap'] for c in compare)}/{len(compare)} saved-time intervals overlap v1",
        "rule": "충족·단축 iff saved-time 95% interval > 0 and recall-loss upper bound < 1%p (v1 rule)",
    })
    pd.set_option("display.width", 300)
    print(df[["split", "family", "policy", "rate", "batch_calls", "total_ms", "saved_ms", "saved_lo", "saved_hi",
              "gru_ms", "rechecks", "recovered_malicious", "new_false_positives", "additional_misses",
              "loss_lo", "loss_hi", "decision"]].round(4).to_string())
    print(pd.DataFrame(compare).round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "dryrun":      # harness check only: 2 repeats into the given scratch folder
        from pathlib import Path
        out = Path(sys.argv[2])
        assert out.resolve() != RESULTS.resolve()
        from v2paths import refuse_protected
        out = refuse_protected(out)
        out.mkdir(parents=True, exist_ok=True)
        measure(2, out)
        summarize(out, out, 2)
    else:
        {"measure": measure, "summarize": summarize}[sys.argv[1]]()
