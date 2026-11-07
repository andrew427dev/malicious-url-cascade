"""Offline CLI for individual URLs and ordered batches with the frozen gate."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.gate import (
    load_policy,
    infer,
    load_models,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--policy-sha256", required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--url")
    inputs.add_argument(
        "--jsonl", type=Path, help="One input_id and raw_url object per line"
    )
    a = p.parse_args()
    policy = load_policy(a.policy, a.policy_sha256)
    t1, t2 = load_models(a.models, policy["models_sha256"])
    if a.url is not None:
        rows = [dict(input_id="1", raw_url=a.url)]
    else:
        with a.jsonl.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    start = perf_counter()
    results = infer(rows, mode="cascade", t1=t1, t2=t2, tau=policy["tau"])
    elapsed = perf_counter() - start
    for row, result in zip(rows, results):
        print(
            json.dumps(
                dict(raw_url=row["raw_url"], **result),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    print(
        json.dumps(dict(batch_rows=len(rows), elapsed_seconds=elapsed)), file=sys.stderr
    )


if __name__ == "__main__":
    main()
