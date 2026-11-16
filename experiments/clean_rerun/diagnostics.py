"""Trace representative errors and audit a preserved evaluation-only revision."""

import json
import numpy as np
import pandas as pd
from audit_results import mask_for
from common import ROOT, sha, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def saved_scores(folder):
    """Only load this run's hash-checked arrays, whose saved IDs have object dtype."""
    complete = read(folder / "complete.json")
    assert complete["status"] == "PASS"
    path = folder / "test-scores.npz"
    assert sha(path) == complete["files"][path.name]
    return np.load(path, allow_pickle=True)


def representative_cases(corpus, kind):
    """Choose the smallest stable ID per failure class, never tune from errors."""
    out = ROOT / kind
    scores = saved_scores(out)
    frozen = read(out / "policy-frozen.json")["families"]
    metrics = read(out / "results.json")
    lookup = corpus.set_index("id")
    y, ids, p2 = scores["y"], scores["id"], scores["p2"]
    prediction = p2 > 0.5
    cases = []
    for family in ("logistic", "boost"):
        p, tau = scores[family], frozen[family]["tau"]
        basic, _ = mask_for(p, ids, tau, 0, "basic")
        boundary, _ = mask_for(p, ids, tau, 0.05, "boundary")
        groups = {
            "early_exit_miss": (y == 1) & prediction & ~boundary,
            "recovered_malicious": (y == 1) & prediction & boundary & ~basic,
            "new_false_positive": (y == 0) & prediction & boundary & ~basic,
            "gru_false_negative": (y == 1) & ~prediction,
        }
        boundary_metrics = metrics["policies"][family + "_boundary"]
        expected = {
            "early_exit_miss": boundary_metrics["additional_misses"],
            "recovered_malicious": boundary_metrics["recovered_malicious"],
            "new_false_positive": boundary_metrics["new_false_positives"],
            "gru_false_negative": metrics["gru"]["fn"],
        }
        for category, mask in groups.items():
            candidates = np.flatnonzero(mask)
            assert len(candidates) == expected[category], (kind, family, category)
            case = {"family": family, "category": category, "population": len(candidates)}
            if len(candidates):
                index = min(candidates, key=lambda i: str(ids[i]))
                row = lookup.loc[ids[index]]
                assert int(row.y_mal) == int(y[index])
                case.update(
                    id=str(ids[index]),
                    truth=int(y[index]),
                    url_characters=len(str(row.url)),
                    t1_malicious_probability=float(p[index]),
                    tau=float(tau),
                    gru_malicious_probability=float(p2[index]),
                    forwarded_basic=bool(basic[index]),
                    forwarded_boundary=bool(boundary[index]),
                )
            cases.append(case)
    return {
        "selection": "Smallest stable ID in each category; illustrative, not a random sample.",
        "scores_sha256": sha(out / "test-scores.npz"),
        "policy_sha256": sha(out / "policy-frozen.json"),
        "cases": cases,
    }


def revision_audit():
    revision = read(ROOT / "evaluation-revision.json")
    assert revision["model_training_repeated"] is False
    assert revision["test_used_for_parameter_selection"] is False
    assert revision["data_manifest_sha256"] == sha(ROOT / "data-final/manifest.json")
    assert revision["training_source_snapshot_sha256"] == sha(ROOT / "source-snapshot.json")
    cohorts = {}
    for kind, record in revision["cohorts"].items():
        out = ROOT / kind
        archive = out / "pre-evaluation-fix"
        complete = read(archive / "complete.json")
        for name, digest in record["archived_evaluation_files"].items():
            assert sha(archive / name) == digest
        for name, digest in record["training_files"].items():
            assert sha(out / name) == digest == complete["files"][name]
        seal = read(out / "training-manifest.json")
        assert seal["files"] == record["training_files"]
        assert seal["prior_complete_sha256"] == sha(archive / "complete.json")
        before, after = saved_scores(archive), saved_scores(out)
        for key in ("y", "id", "logistic", "boost"):
            assert np.array_equal(before[key], after[key]), (kind, key)
        # The exact float32 benign model output is recoverable from the new float64 score.
        benign = (1.0 - after["p2"]).astype(np.float32)
        old_conversion = np.float32(1.0) - benign
        assert np.array_equal(before["p2"], old_conversion)
        assert np.array_equal(after["p2"] > 0.5, benign < 0.5)
        old_policy, new_policy = (
            read(folder / "policy-frozen.json")["families"] for folder in (archive, out)
        )
        cohorts[kind] = {
            "training_files_unchanged": "PASS",
            "test_rows_and_t1_scores_unchanged": "PASS",
            "float_conversion_reproduced": "PASS",
            "test_score_values_changed": int(np.sum(before["p2"] != after["p2"])),
            "test_gru_decisions_changed": int(np.sum((before["p2"] > 0.5) != (after["p2"] > 0.5))),
            "tau_before": {k: v["tau"] for k, v in old_policy.items()},
            "tau_after": {k: v["tau"] for k, v in new_policy.items()},
            "corrected_complete_sha256": sha(out / "complete.json"),
        }
    return {
        "status": "PASS",
        "revision_sha256": sha(ROOT / "evaluation-revision.json"),
        "cohorts": cohorts,
    }


def main():
    assert read(ROOT / "result-audit.json")["status"] == "PASS"
    corpus = pd.read_csv(
        ROOT / "data-final/clean.csv", usecols=["id", "y_mal", "url"], keep_default_na=False
    )
    result = {
        "status": "PASS",
        "auditor_sha256": sha(__file__),
        "result_audit_sha256": sha(ROOT / "result-audit.json"),
        "error_cases": {kind: representative_cases(corpus, kind) for kind in ("random", "domain")},
    }
    if (ROOT / "evaluation-revision.json").exists():
        result["evaluation_revision"] = revision_audit()
    write_json(ROOT / "diagnostics.json", result)
    print("DIAGNOSTICS PASS; descriptive only, no model or policy changes")


if __name__ == "__main__":
    main()
