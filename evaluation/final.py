"""Evaluate the frozen B policy without changing either model or the gate."""

import argparse
from collections import defaultdict
import ipaddress
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.gate import (
    load_policy,
    SPLIT_HASH,
    binary_metrics,
    cascade_invariants,
    compute_context,
    infer,
    load_models,
    load_rows,
    predictions,
    read_json,
    verify_file,
    write_json,
)
from urlcascade.benchmark import fixed_sample, sequential_benchmark
from urlcascade.core import hostname_for_grouping

EXTERNAL_SHA256 = "6bcb5a84b843675eef70e23e867c6101ee1a669dfb1d885a2b499c9a25e07e90"
MODES = ("always_normal", "t1", "t2", "cascade")


def paired_intervals(rows, base, cascade, repeats=2000):
    import numpy as np

    groups = defaultdict(lambda: [0, 0, 0])
    for row, b, c in zip(rows, base, cascade):
        total = groups[row["group_id"]]
        if row["y_mal"]:
            total[0] += 1
            total[1] += b
            total[2] += c
    grouped = np.array([groups[g] for g in sorted(groups)], dtype=np.int64)
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(repeats):
        m, b, c = grouped[rng.integers(len(grouped), size=len(grouped))].sum(axis=0)
        if m:
            estimates.append([b / m, c / m, (b - c) / m])
    names = ("t2_recall", "cascade_recall", "recall_drop")
    ranges = (
        np.quantile(estimates, [0.025, 0.975], axis=0).T if estimates else [None] * 3
    )
    return dict(
        seed=42,
        repeats=repeats,
        valid=len(estimates),
        reliable=len(estimates) >= 1900,
        intervals={k: None if v is None else v.tolist() for k, v in zip(names, ranges)},
    )


def throughput_benchmark(rows, t1, t2, tau):
    sample = fixed_sample(rows, 10000)
    rng = random.Random(42)
    results = []

    def run(block, mode):
        return infer(block, mode=mode, t1=t1, t2=t2, tau=tau)

    for mode in MODES:
        run(sample[:100], mode)
    for repeat in range(5):
        order = list(MODES)
        rng.shuffle(order)
        for position, mode in enumerate(order):
            elapsed, calls = [], 0
            for offset in range(0, len(sample), 128):
                block = sample[offset : offset + 128]
                start = time.perf_counter()
                outputs = run(block, mode)
                elapsed.append(time.perf_counter() - start)
                if any(r["status"] != "ok" for r in outputs):
                    raise ValueError("Invalid row in throughput sample")
                calls += sum(r["t2_called"] for r in outputs)
            results.append(
                dict(
                    repetition=repeat,
                    order_index=position,
                    mode=mode,
                    t2_calls=calls,
                    batch_seconds=elapsed,
                    rows_per_second=len(sample) / sum(elapsed),
                )
            )
    return dict(
        batch_size=128, sample_ids=[r["input_id"] for r in sample], measurements=results
    )


def subgroup_counts(rows, baseline, cascade, calls, cuts):
    from bisect import bisect_left

    groups = defaultdict(lambda: [0, 0, 0, 0])
    for r, b, c, called in zip(rows, baseline, cascade, calls):
        raw = r["raw_url"]
        try:
            ipaddress.ip_address(hostname_for_grouping(raw))
            ip = True
        except ValueError:
            ip = False
        explicit_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw))
        source = r["source"]
        if isinstance(source, list):
            source = "+".join(source)
        for key in (
            f"length_Q{bisect_left(cuts, len(raw)) + 1}",
            f"ip_host={ip}",
            f"has_scheme={explicit_scheme}",
            "source=" + source,
        ):
            g = groups[key]
            g[0] += 1
            g[1] += r["y_mal"]
            g[2] += int(r["y_mal"] and b and not c)
            g[3] += called
    return {
        k: dict(rows=n, malicious=m, additional_fn=fn, t2_calls=calls)
        for k, (n, m, fn, calls) in sorted(groups.items())
    }


def evaluate(rows, t1, t2, tau, cuts, out):
    out.mkdir()
    labels = [r["y_mal"] for r in rows]
    decisions, scores = {}, {}
    with (out / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for mode in MODES:
            outputs = predictions(rows, mode=mode, t1=t1, t2=t2, tau=tau)
            decisions[mode] = [r["label_mal"] for r in outputs]
            scores[mode] = binary_metrics(labels, decisions[mode])
            if mode == "cascade":
                calls = [r["t2_called"] for r in outputs]
            for result, y in zip(outputs, labels):
                stream.write(
                    json.dumps(dict(mode=mode, y_mal=y, **result), allow_nan=False)
                    + "\n"
                )
    base, casc = decisions["t2"], decisions["cascade"]
    checks = cascade_invariants(labels, base, casc, calls)
    if any(c != (b if called else 0) for b, c, called in zip(base, casc, calls)):
        raise ValueError("Forwarded decisions differ from the T2 baseline")
    per_request = sequential_benchmark(rows, t1=t1, t2=t2, thresholds=[tau])
    write_json(out / "sequential-timings.json", per_request)
    batches = throughput_benchmark(rows, t1, t2, tau)
    write_json(out / "batch-timings.json", batches)
    timing = {}
    for mode in MODES:
        repeats = [m for m in per_request["measurements"] if m["mode"] == mode]
        means = [m["mean_seconds"] for m in repeats]
        timing[mode] = dict(
            batch1_median_mean_s=statistics.median(means),
            batch1_mean_range_s=[min(means), max(means)],
            batch1_median_p95_s=statistics.median(m["p95_seconds"] for m in repeats),
            batch128_median_rows_per_s=statistics.median(
                m["rows_per_second"]
                for m in batches["measurements"]
                if m["mode"] == mode
            ),
        )
    latency_ratio = (
        timing["cascade"]["batch1_median_mean_s"] / timing["t2"]["batch1_median_mean_s"]
    )
    early = len(rows) - sum(calls)
    malicious = sum(labels)
    base_recall = scores["t2"]["recall"]
    return dict(
        goals=dict(
            recall_loss_at_most_one_point=checks["allowed"] if malicious else None,
            call_reduction_at_least_30_percent=(10 * early >= 3 * len(rows)),
            latency_reduction_at_least_20_percent=latency_ratio <= 0.8,
        ),
        recall_retention=(
            scores["cascade"]["recall"] / base_recall if base_recall else None
        ),
        early_exits=early,
        early_malicious_share_of_malicious=(
            checks["early_malicious"] / malicious if malicious else None
        ),
        early_malicious_share_of_exits=(
            checks["early_malicious"] / early if early else None
        ),
        rows=len(rows),
        groups=len({r["group_id"] for r in rows}),
        malicious=sum(labels),
        normal=len(labels) - sum(labels),
        metrics=scores,
        comparison=checks,
        tau=tau,
        timing=timing,
        speed_sample_rows=len(per_request["sample_ids"]),
        throughput_sample_rows=len(batches["sample_ids"]),
        latency_reduction=1 - latency_ratio,
        call_reduction=1 - checks["call_rate"],
        speedup=1 / latency_ratio,
        bootstrap=paired_intervals(rows, base, casc),
        train_length_quartiles=cuts,
        subgroups=subgroup_counts(rows, base, casc, calls, cuts),
    )


def main():

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "split", "external", "policy", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--policy-sha256", required=True)
    a = p.parse_args()
    import numpy as np

    compute_context("kaggle", a.out)
    policy = load_policy(a.policy, a.policy_sha256)
    verify_file(a.split / "manifest.json", SPLIT_HASH)
    split = read_json(a.split / "manifest.json")
    verify_file(a.split / "train.jsonl", split["files"]["train.jsonl"])
    with (a.split / "train.jsonl").open() as stream:
        cuts = np.quantile(
            [len(json.loads(line)["raw_url"]) for line in stream], [0.25, 0.5, 0.75]
        ).tolist()
    load_start = time.perf_counter()
    t1, t2 = load_models(a.models, policy["models_sha256"])
    model_load_seconds = time.perf_counter() - load_start
    a.out.mkdir(parents=True, exist_ok=False)
    for name, path, digest in [
        ("test", a.split / "test.jsonl", split["files"]["test.jsonl"]),
        ("external", a.external, EXTERNAL_SHA256),
    ]:
        read_start = time.perf_counter()
        rows = load_rows(path, digest)
        dataset_read_seconds = time.perf_counter() - read_start
        report = evaluate(rows, t1, t2, policy["tau"], cuts, a.out / name)
        report.update(
            dataset=name,
            model_load_verify_seconds=model_load_seconds,
            dataset_read_verify_seconds=dataset_read_seconds,
            dataset_sha256=digest,
            policy_sha256=a.policy_sha256,
            models_sha256=policy["models_sha256"],
        )
        write_json(a.out / name / "report.json", report)
        print(name, json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
