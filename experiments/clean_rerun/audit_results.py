"""Recalculate saved predictions and policy outcomes without importing the runner."""

import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from common import ROOT, sha, write_json, verify_execution_source


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def mask_for(prob, ids, tau, rate, policy, seed=0):
    forwarded = prob >= tau
    rows = np.flatnonzero(~forwarded).tolist()
    budget = math.floor(len(rows) * rate)
    if policy == "random":
        rows.sort(
            key=lambda i: (
                hashlib.sha256((str(seed) + "|" + str(ids[i])).encode("utf-8")).digest(),
                str(ids[i]),
            )
        )
    elif policy == "boundary":
        rows.sort(key=lambda i: (-prob[i], str(ids[i])))
    elif policy != "basic":
        raise ValueError(policy)
    forwarded[rows[:budget]] = True
    return forwarded, budget


def check(y, p2, mask, record, base=None):
    pred = (p2 > 0.5) & mask
    ben = y == 0
    mal = y == 1
    counts = {
        "tn": int(np.sum(ben & ~pred)),
        "fp": int(np.sum(ben & pred)),
        "fn": int(np.sum(mal & ~pred)),
        "tp": int(np.sum(mal & pred)),
    }
    for k, v in counts.items():
        assert v == record[k], (k, v, record[k])
    tp, fp, fn, tn = [counts[k] for k in ("tp", "fp", "fn", "tn")]
    values = {
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "fpr": fp / (fp + tn),
    }
    for k, v in values.items():
        assert abs(v - record[k]) < 1e-12, (k, v, record[k])
    missed = int(np.sum(mal & (p2 > 0.5) & ~mask))
    assert missed == record["additional_misses"]
    assert abs(missed / mal.sum() - record["recall_loss"]) < 1e-12
    assert int(mask.sum()) == record["t2_calls"]
    assert abs(float(mask.mean()) - record["t2_call_rate"]) < 1e-12
    assert fp <= int(np.sum(ben & (p2 > 0.5)))
    if base is not None:
        assert np.all(mask[base])
        assert int(np.sum((mask & ~base) & mal & (p2 > 0.5))) == record["recovered_malicious"]
        assert int(np.sum((mask & ~base) & ben & (p2 > 0.5))) == record["new_false_positives"]


def main():
    result = {"status": "PASS", "cohorts": {}}
    corpus = pd.read_csv(
        ROOT / "data-final/clean.csv", usecols=["id", "y_mal"], keep_default_na=False
    )
    for kind in ("random", "domain"):
        p = ROOT / kind
        complete = read(p / "complete.json")
        assert complete["status"] == "PASS"
        for name, digest in complete["files"].items():
            assert sha(p / name) == digest, name
        settings = read(p / "settings.json")
        assert settings["test_used_for_selection"] is False
        assert settings["protocol_sha256"] == sha(ROOT / "protocol.json")
        assert settings["protocol"] == read(ROOT / "protocol.json")
        assert settings["protocol"]["prior_test_results_seen"] is True
        assert settings["protocol"]["test_used_for_parameter_selection"] is False
        partition_indices = np.load(ROOT / "data-final" / (kind + "_indices.npz"))
        shared_validation = np.sort(
            np.concatenate([partition_indices[k] for k in ("model", "calibration", "policy")])
        )
        assert settings["validation_rows"] == len(shared_validation)
        assert (
            settings["validation_indices_sha256"]
            == hashlib.sha256(shared_validation.tobytes()).hexdigest()
        )
        assert not np.intersect1d(shared_validation, partition_indices["test"]).size
        assert not np.intersect1d(shared_validation, partition_indices["train"]).size
        verify_execution_source("experiment.py", settings["experiment_sha256"])
        verify_execution_source("experiment.py", read(p / "evaluation-code.json")["code_sha256"])
        assert settings["data_manifest_sha256"] == sha(ROOT / "data-final/manifest.json")
        for name, digest in settings["source_files_sha256"].items():
            verify_execution_source(name, digest, reference=True)
        policy = read(p / "policy-frozen.json")
        assert policy["frozen_before_test"] is True
        for name, digest in policy["model_hashes"].items():
            assert sha(p / name) == digest
        r = read(p / "results.json")
        s = np.load(p / "test-scores.npz", allow_pickle=True)
        y = s["y"]
        p2 = s["p2"]
        ids = s["id"]
        assert len(set(ids)) == len(ids)
        test_indices = np.load(ROOT / "data-final" / (kind + "_indices.npz"))["test"]
        assert np.array_equal(ids, corpus.id.iloc[test_indices].to_numpy())
        assert np.array_equal(y, corpus.y_mal.iloc[test_indices].to_numpy())
        assert set(y) == {0, 1}
        check(y, p2, np.ones(len(y), dtype=bool), r["gru"])
        assert abs(average_precision_score(y, p2) - r["gru"]["pr_auc_ap"]) < 1e-12
        assert abs(roc_auc_score(y, p2) - r["gru"]["roc_auc"]) < 1e-12
        checks = 1
        for family in ("logistic", "boost"):
            probabilities = s[family]
            tau = policy["families"][family]["tau"]
            eligible = [
                v for v in policy["families"][family]["candidates"] if v["recall_loss"] <= 0.01
            ]
            assert min(eligible, key=lambda v: (v["t2_call_rate"], v["tau"]))["tau"] == tau
            base, _ = mask_for(probabilities, ids, tau, 0, "basic")
            for mode, rate in [("basic", 0), ("random", 0.05), ("boundary", 0.05)]:
                mask, budget = mask_for(probabilities, ids, tau, rate, mode)
                row = r["policies"][family + "_" + mode]
                check(y, p2, mask, row, base)
                assert budget == row["rechecked"]
                assert (
                    abs(average_precision_score(y, np.where(mask, p2, -1.0)) - row["pr_auc_ap"])
                    < 1e-12
                )
                assert abs(roc_auc_score(y, np.where(mask, p2, -1.0)) - row["roc_auc"]) < 1e-12
                checks += 1
            for row in r["random_seed_summary"][family]["records"]:
                mask, budget = mask_for(probabilities, ids, tau, 0.05, "random", row["seed"])
                check(y, p2, mask, row, base)
                checks += 1
            seeds = r["random_seed_summary"][family]
            assert [v["seed"] for v in seeds["records"]] == list(range(30))
            for metric in seeds["mean"]:
                samples = [v[metric] for v in seeds["records"]]
                assert abs(float(np.mean(samples)) - seeds["mean"][metric]) < 1e-12
                assert abs(float(np.std(samples, ddof=1)) - seeds["std"][metric]) < 1e-12
            for row in r["sensitivity"][family]:
                mask, budget = mask_for(probabilities, ids, row["tau"], row["rate"], row["policy"])
                baseline = probabilities >= row["tau"]
                check(y, p2, mask, row, baseline)
                assert row["rechecked"] == budget
                checks += 1
            for rate in (0.01, 0.05, 0.1, 0.2):
                for threshold in settings["taus"]:
                    pair = [
                        row
                        for row in r["sensitivity"][family]
                        if row["tau"] == threshold and row["rate"] == rate
                    ]
                    assert len(pair) == 2 and pair[0]["rechecked"] == pair[1]["rechecked"]
        tm = read(p / "timing.json")
        assert tm["warmups_per_policy"] == 1 and tm["repeats"] == 5
        positions = {v: i for i, v in enumerate(ids)}
        ix = np.array([positions[v] for v in tm["sample_ids"]])
        assert len(ix) == 1000 and len(set(ix)) == 1000
        for name, rounds in tm["rounds"].items():
            assert len(rounds) == 5
            assert sorted(row["repeat"] for row in rounds) == list(range(5))
            if name == "gru":
                forward = np.ones(len(ix), dtype=bool)
            else:
                family, mode = name.split("_")
                forward, _ = mask_for(
                    s[family][ix],
                    ids[ix],
                    policy["families"][family]["tau"],
                    0 if mode == "basic" else 0.05,
                    mode,
                )
            expected_hash = hashlib.sha256(((p2[ix] > 0.5) & forward).tobytes()).hexdigest()
            for row in rounds:
                assert row["predictions_sha256"] == expected_hash, (kind, name, "actual_prediction")
                assert row["t2_calls"] == int(forward.sum())
                assert row["n"] == len(ix)
                assert (
                    abs(
                        sum(row[k] for k in ("feature_s", "t1_s", "routing_s", "t2_s"))
                        - row["total_s"]
                    )
                    < 1e-8
                )
                assert row["total_s"] > 0
            summary = tm["summary"][name]
            for part in ("feature", "t1", "routing", "t2", "total"):
                assert (
                    abs(
                        summary["mean_" + part + "_s"]
                        - np.mean([row[part + "_s"] for row in rounds])
                    )
                    < 1e-12
                )
            assert (
                abs(summary["median_total_s"] - np.median([row["total_s"] for row in rounds]))
                < 1e-12
            )
            assert (
                abs(
                    summary["amortized_ms_per_url"]
                    - np.median([row["total_s"] / row["n"] * 1000 for row in rounds])
                )
                < 1e-12
            )
            assert (
                abs(
                    summary["throughput_urls_s"]
                    - np.median([row["n"] / row["total_s"] for row in rounds])
                )
                < 1e-8
            )
            assert summary["t2_calls"] == int(forward.sum())
        result["cohorts"][kind] = {
            "policy_metric_records_checked": checks,
            "timing_rounds_checked": 35,
            "same_budget": "PASS",
            "actual_selective_predictions": "PASS",
            "artifact_hashes": "PASS",
            "test_identity_and_labels": "PASS",
            "seed_and_timing_aggregates": "PASS",
            "shared_validation_rows": len(shared_validation),
            "prior_test_exposure_recorded": "PASS",
            "test_class_counts": {"benign": int((y == 0).sum()), "malicious": int((y == 1).sum())},
        }
    curves = read(ROOT / "learning-curves.json")
    assert curves["status"] == "PASS"
    verify_execution_source("learning_curves.py", curves["code_sha256"])
    assert curves["data_manifest_sha256"] == sha(ROOT / "data-final/manifest.json")
    assert curves["indices_sha256"] == sha(ROOT / "learning-curve-indices.npz")
    parts = np.load(ROOT / "data-final/random_indices.npz")
    subsets = np.load(ROOT / "learning-curve-indices.npz")
    shared_validation = np.sort(
        np.concatenate([parts[k] for k in ("model", "calibration", "policy")])
    )
    assert curves["validation_rows"] == len(shared_validation)
    assert (
        curves["validation_ids_sha256"]
        == hashlib.sha256("\n".join(corpus.id.iloc[shared_validation]).encode()).hexdigest()
    )
    for family, rows in curves["families"].items():
        previous = set()
        for row in rows:
            ix = subsets[str(row["fraction"])]
            assert previous.issubset(set(ix)) and set(ix).issubset(set(parts["train"]))
            previous = set(ix)
            assert row["training_rows"] == len(ix)
            assert row["training_benign"] == int((corpus.y_mal.iloc[ix] == 0).sum())
            assert row["training_malicious"] == int((corpus.y_mal.iloc[ix] == 1).sum())
            assert (
                row["training_ids_sha256"]
                == hashlib.sha256("\n".join(corpus.id.iloc[ix]).encode()).hexdigest()
            )
        selected = read(ROOT / "random/t1-selection.json")["candidates"][family]
        anchor = next(
            v["model_validation_ap"]
            for v in selected["entries"]
            if v["setting"] == selected["winner"]
        )
        assert abs(rows[-1]["validation_ap"] - anchor) < 1e-10
    result["learning_curves"] = {
        "status": "PASS",
        "fit_records": 10,
        "train_only_subsets": "PASS",
        "full_size_anchor": "PASS",
    }
    result["auditor_sha256"] = sha(__file__)
    write_json(ROOT / "result-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
