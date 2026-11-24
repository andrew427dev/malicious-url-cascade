"""Benchmark real selective execution on an identical raw batch for both forms."""

import argparse
import time
from inputs import (
    ROOT,
    PREFIX,
    np,
    pd,
    tf,
    vectorizer,
    preprocess_link,
    extract_features,
    write_json,
)
from evaluate import FAMILIES, POLICIES, load_gru, load_stages, load_rows, route, read


def prepare_benchmark(kind):
    out = ROOT / kind
    group, form = kind.split("-")
    frame, y, tok, x = load_rows(out, "test")
    other = "original" if form == "stripped" else "stripped"
    opposite = pd.read_csv(ROOT / (group + "-" + other) / "test-rows.csv.gz").id
    common = frame.index.intersection(opposite).sort_values()
    if len(common) < 1000:
        raise ValueError("Too few shared cases for timing")
    rng = np.random.default_rng(46)
    ids = rng.choice(common, 1000, replace=False)
    index = frame.index.get_indexer(ids)
    raw = frame.url.iloc[index].tolist()
    layer = vectorizer(read(out / "vocab.json"))
    stages = load_stages(out)
    model = load_gru(out)
    frozen = read(out / "routing.json")
    from inputs import sha

    if sha(out / "gru.h5") != frozen["gru_sha256"] or any(
        sha(out / "stage1" / (f + ".joblib")) != frozen["first_stage_sha256"][f]
        for f in FAMILIES
    ):
        raise ValueError("Frozen routing/model mismatch")
    taus = {f: frozen["families"][f]["tau"] for f in FAMILIES}
    with np.load(out / "scores.npz") as stored:
        assert np.array_equal(ids, stored["id"][index]) and np.array_equal(
            y[index], stored["y"][index]
        )
        full = stored["gru"][index] > 0.5
        expected = {"gru": np.ones(1000, bool)}
        for f in FAMILIES:
            for p, r in POLICIES:
                expected[f + "_" + p] = route(stored[f][index], ids, taus[f], r, p)[0]

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 50], tf.int32),
            tf.TensorSpec([None, 18], tf.float32),
        ]
    )
    def forward(tokens, numeric):
        return model([tokens, numeric], training=False)

    def execute(name):
        start = time.perf_counter_ns()
        urls = [PREFIX.sub("", u, count=1) for u in raw] if form == "stripped" else raw
        numeric = np.asarray([extract_features(u) for u in urls], dtype=np.float32)
        feature_end = time.perf_counter_ns()
        if name == "gru":
            mask = np.ones(len(raw), bool)
        else:
            family, policy = name.split("_")
            p = stages[family].predict_proba(numeric)[:, 1]
            mask, _ = route(
                p, ids, taus[family], 0 if policy == "basic" else 0.05, policy
            )
        route_end = time.perf_counter_ns()
        chosen = np.flatnonzero(mask)
        prediction = np.zeros(len(raw), bool)
        if len(chosen):
            tokens = tf.cast(
                layer(tf.constant([preprocess_link(urls[i]) for i in chosen])), tf.int32
            )
            prediction[chosen] = (
                forward(tokens, numeric[chosen]).numpy().reshape(-1) < 0.5
            )
        end = time.perf_counter_ns()
        np.testing.assert_array_equal(numeric, x[index])
        np.testing.assert_array_equal(mask, expected[name])
        np.testing.assert_array_equal(prediction, full & expected[name])
        return {
            "name": name,
            "calls": int(mask.sum()),
            "total_ms": (end - start) / 1e6,
            "features_ms": (feature_end - start) / 1e6,
            "route_ms": (route_end - feature_end) / 1e6,
            "gru_ms": (end - route_end) / 1e6,
        }

    return ids.tolist(), len(common), expected, execute


def save_timing(kind, ids, common_n, expected, runs):
    from statistics import median

    out = ROOT / kind
    summary = {}
    for name in expected:
        rows = [r for r in runs if r["name"] == name]
        times = [r["total_ms"] for r in rows]
        summary[name] = {
            "ms": median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "calls": rows[0]["calls"],
        }
    for row in summary.values():
        row["time_saved_pct"] = 100 * (1 - row["ms"] / summary["gru"]["ms"])
        row["calls_saved_pct"] = 100 * (1 - row["calls"] / len(ids))
    write_json(
        out / "timing.json",
        {
            "sample_ids": ids,
            "common_n": common_n,
            "batch": len(ids),
            "warmup": 1,
            "repeats": 5,
            "threads": 1,
            "rounds": runs,
            "summary": summary,
        },
    )


def benchmark(group):
    forms = ("original", "stripped")
    jobs = {}
    for form in forms:
        kind = group + "-" + form
        if (ROOT / kind / "timing.json").exists():
            raise FileExistsError("Existing timing")
        jobs[form] = prepare_benchmark(kind)
    assert jobs["original"][0] == jobs["stripped"][0]
    grid = [(f, n) for f in forms for n in jobs[f][2]]
    for form, name in grid:
        jobs[form][3](name)
    runs = {f: [] for f in forms}
    rng = np.random.default_rng(47)
    for repeat in range(5):
        for pair_order, i in enumerate(rng.permutation(len(grid)), 1):
            form, name = grid[i]
            runs[form].append(
                {
                    "repeat": repeat + 1,
                    "pair_order": pair_order,
                    "order": 1 + len(runs[form]) % 7,
                    **jobs[form][3](name),
                }
            )
    for form in forms:
        ids, n, expected, _ = jobs[form]
        save_timing(group + "-" + form, ids, n, expected, runs[form])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    benchmark(parser.parse_args().kind)
