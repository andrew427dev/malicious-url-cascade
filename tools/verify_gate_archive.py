"""Verify a W6 result against its Kaggle log and print aggregate results for W7."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

from evaluation.gate import policy_code, verify_file
from urlcascade.core import GATE_CANDIDATES, select_gate

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_timings(timings, allowed, n):
    ids = timings["sample_ids"]
    require(
        timings["batch_size"] == 1 and len(ids) == len(set(ids)) == min(n, 1000),
        "Unexpected batch size or timing sample",
    )
    paths = {("always_normal", None), ("t1", None), ("t2", None)}
    paths.update(("cascade", tau) for tau in allowed)
    means = {}
    positions = set()
    sample_calls = {}
    for row in timings["measurements"]:
        path = (row["mode"], row["tau"])
        rep, pos = row["repetition"], row["order_index"]
        key = (path, rep)
        require(
            path in paths and rep in range(5) and key not in means,
            "Unexpected or duplicate timing repetition",
        )
        require(
            pos in range(len(paths)) and (rep, pos) not in positions,
            "Unexpected or duplicate path order",
        )
        positions.add((rep, pos))
        raw = row["raw_seconds"]
        require(
            row["n"] == len(raw) == len(ids)
            and all(math.isfinite(t) and t >= 0 for t in raw),
            "Invalid raw timings",
        )
        mean = statistics.mean(raw)
        require(
            mean > 0 and math.isclose(mean, row["mean_seconds"], rel_tol=1e-12),
            "Raw and reported mean latency differ",
        )
        calls = row["t2_calls"]
        expected = {"always_normal": 0, "t1": 0, "t2": len(ids)}.get(path[0])
        if path == ("cascade", 0):
            expected = len(ids)
        require(
            type(calls) is int
            and 0 <= calls <= len(ids)
            and (expected is None or calls == expected)
            and sample_calls.setdefault(path, calls) == calls,
            "Unexpected sample T2 calls",
        )
        means[key] = row["mean_seconds"]
    require(len(means) == 5 * len(paths), "Missing timing repetitions")
    return {p: [means[p, r] for r in range(5)] for p in paths}


def check_candidates(candidates, policy, timings, partition):
    n, m = partition["rows"], partition["malicious"]
    require(
        [c["tau"] for c in candidates] == list(GATE_CANDIDATES),
        "Unexpected candidate grid",
    )
    baseline = candidates[0]["metrics"]
    checked = []
    last_calls, last_tp, last_fp = n, m, n - m
    for c in candidates:
        metrics = c["metrics"]
        tp, fn, tn, fp = (metrics[k] for k in ("tp", "fn", "tn", "fp"))
        require(
            all(type(x) is int and x >= 0 for x in (tp, fn, tn, fp))
            and tp + fn == m
            and tn + fp == n - m
            and metrics["n"] == n,
            "Candidate confusion counts do not match the frozen cohort",
        )
        k = baseline["tp"] - tp
        rate = c["call_rate"]
        calls = round(rate * n)
        require(
            tp <= last_tp
            and fp <= last_fp
            and tp + fp <= calls <= last_calls
            and math.isclose(rate, calls / n, rel_tol=0, abs_tol=1e-12)
            and (c["tau"] != 0 or calls == n),
            "Invalid T2 routing counts",
        )
        last_calls, last_tp, last_fp = calls, tp, fp
        allowed = 100 * k <= m
        require(
            c["malicious"] == m
            and c["additional_fn"] == k
            and math.isclose(c["recall_drop"], k / m, abs_tol=1e-12)
            and math.isclose(metrics["recall"], tp / m, abs_tol=1e-12)
            and c["allowed"] is allowed,
            "Additional misses or 1 percentage point rule differ",
        )
        checked.append(
            dict(
                tau=c["tau"],
                additional_fn=k,
                malicious=m,
                recall_drop=k / m,
                t2_calls=calls,
                call_rate=rate,
                allowed=allowed,
            )
        )
    means = check_timings(timings, [c["tau"] for c in checked if c["allowed"]], n)
    for item, c in zip(checked, candidates):
        times = means.get(("cascade", c["tau"]), [])
        require(
            c["mean_latencies"] == times, "Candidate times differ from timing evidence"
        )
        item["mean_latencies"] = times
    tau = select_gate(checked)
    chosen = next(c for c in checked if c["tau"] == tau)
    require(
        policy["gate_rows"] == n
        and all(
            policy[k] == chosen[k]
            for k in ("tau", "additional_fn", "malicious", "recall_drop", "call_rate")
        ),
        "Frozen policy disagrees with candidate selection",
    )
    require(
        policy["t2_median_mean_seconds"] == statistics.median(means[("t2", None)])
        and policy["cascade_median_mean_seconds"]
        == statistics.median(chosen["mean_latencies"]),
        "Frozen policy latency disagrees with timing evidence",
    )
    return checked


def verify_archive(path, archive_sha256, policy_sha256):
    verify_file(path, archive_sha256)
    model = json.loads((ROOT / "evidence/experiment-b-selection.json").read_text())
    split = json.loads((ROOT / "evidence/b-split-manifest.json").read_text())
    with zipfile.ZipFile(path) as archive:
        require(
            not any(n > 1 for n in Counter(archive.namelist()).values()),
            "Duplicate ZIP member",
        )
        require(archive.testzip() is None, "ZIP CRC mismatch")
        require(
            archive.read("evaluation/policy.sha256").decode().strip() == policy_sha256,
            "Archive policy digest differs from the Kaggle log",
        )

        def read_report(name, digest):
            data = archive.read("evaluation/gate/" + name)
            require(
                hashlib.sha256(data).hexdigest() == digest,
                f"Checksum mismatch for {name}",
            )
            return json.loads(data)

        policy = read_report("policy.json", policy_sha256)
        require(
            policy["experiment"] == "B"
            and policy["stage"] == "gate_selection"
            and policy["models_sha256"] == model["model_manifest_sha256"]
            and policy["split_sha256"] == model["split_manifest_sha256"]
            and policy["gate_file_sha256"] == split["files"]["gate_selection.jsonl"]
            and policy["code_sha256"] == policy_code(),
            "Frozen model, split or code mismatch",
        )
        evidence = policy["evidence_sha256"]
        candidates = read_report(
            "gate-candidates.json", evidence["gate-candidates.json"]
        )
        timings = read_report("gate-latencies.json", evidence["gate-latencies.json"])
    rows = check_candidates(
        candidates, policy, timings, split["summary"]["gate_selection"]
    )
    return dict(
        archive_sha256=archive_sha256,
        policy_sha256=policy_sha256,
        gate_rows=policy["gate_rows"],
        tau=policy["tau"],
        candidates=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--archive-sha256",
        required=True,
        help="Verified gate archive digest from the W6 log",
    )
    parser.add_argument(
        "--policy-sha256", required=True, help="Frozen B policy digest from the W6 log"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_archive(args.archive, args.archive_sha256, args.policy_sha256),
            indent=2,
        )
    )
