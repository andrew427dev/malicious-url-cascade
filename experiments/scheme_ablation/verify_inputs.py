"""Check row identity and the shared PR #3 artifacts without loading a trained model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clean_rerun"))
from common import ROOT, sha, validation_indices, verify_training_artifacts

EVIDENCE = HERE.parents[1] / "evidence"
BASELINE = EVIDENCE / "clean-rerun-shared-validation"
ROW_REFERENCE = EVIDENCE / "scheme-ablation" / "pr3-ordered-cohorts.json"
KINDS = ("random", "domain")
ROLES = ("train", "model", "calibration", "policy", "test")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files(data):
    expected = read(BASELINE / "data-final" / "manifest.json")["files"]
    actual = read(data / "manifest.json")["files"]
    if actual.keys() != expected.keys():
        raise ValueError("data_file_list_changed")
    for name, digest in actual.items():
        if sha(data / name) != digest:
            raise ValueError("data_file_changed:" + name)
    return {name: actual[name] == expected[name] for name in expected}


def checked_parts(data, df, kind):
    assignments = df[kind + "_split"].to_numpy()
    with np.load(data / (kind + "_indices.npz"), allow_pickle=False) as archive:
        if set(archive.files) != set(ROLES) or not df[kind + "_split"].isin(ROLES).all():
            raise ValueError("invalid_split_roles:" + kind)
        parts = {role: archive[role] for role in ROLES}
    for role, rows in parts.items():
        if rows.dtype.kind not in "iu" or not np.array_equal(rows, np.flatnonzero(assignments == role)):
            raise ValueError("split_index_mismatch:" + kind + ":" + role)
    return parts


def ordered_cohorts(data, df):
    if (not df.id.is_unique or not df.id.str.fullmatch(r"[0-9a-f]{64}").all()
            or not df.y_mal.isin([0, 1]).all()):
        raise ValueError("invalid_id_or_label")
    records = [f"{key}\t{label}\n".encode("ascii") for key, label in zip(df.id, df.y_mal)]
    result = {}
    for kind in KINDS:
        parts = checked_parts(data, df, kind)
        parts["shared_validation"] = validation_indices(parts)
        result[kind] = {}
        for role, rows in parts.items():
            digest = hashlib.sha256()
            for idx in rows:
                digest.update(records[idx])
            result[kind][role] = {"rows": len(rows), "sha256": digest.hexdigest()}
    return result


def verified_rows(data, **csv_options):
    matches = verify_files(data)
    for name in ("random-encoded.npy", "domain-encoded.npy", "random-vocabulary.json",
                 "domain-vocabulary.json", "normalization-quarantine.csv"):
        if not matches[name]:
            raise ValueError("baseline_input_changed:" + name)
    df = pd.read_csv(data / "clean.csv", keep_default_na=False, **csv_options)
    if ordered_cohorts(data, df) != read(ROW_REFERENCE):
        raise ValueError("ordered_cohorts_differ_from_pr3")
    return df, matches


def verify_shared_run(root, kind):
    out = root / kind
    expected = read(BASELINE / kind / "complete.json")
    if read(out / "complete.json") != expected or expected["kind"] != kind or expected["status"] != "PASS":
        raise ValueError("completion_differs_from_pr3:" + kind)
    verify_training_artifacts(out)
    for name in ("settings.json", "training-manifest.json", "policy-frozen.json"):
        if sha(out / name) != expected["files"][name]:
            raise ValueError("completed_artifact_changed:" + name)
    settings = read(out / "settings.json")
    if settings["kind"] != kind or settings["data_manifest_sha256"] != sha(root / "data-final" / "manifest.json"):
        raise ValueError("training_data_mismatch:" + kind)
    policy = read(out / "policy-frozen.json")
    models = ("t2.h5", "vocabulary.json", "logistic.joblib", "boost.joblib")
    if policy["frozen_before_test"] is not True or policy["model_hashes"] != {
        name: expected["files"][name] for name in models
    }:
        raise ValueError("policy_model_mismatch:" + kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-models", action="store_true", help="also check both original PR #3 model and policy bundles")
    args = parser.parse_args()
    df, matches = verified_rows(ROOT / "data-final", usecols=["id", "y_mal", "random_split", "domain_split"])
    if args.shared_models:
        for kind in KINDS:
            verify_shared_run(ROOT, kind)
    print(json.dumps({"rows": len(df), "ordered_cohorts": "PASS", "data_hash_equal_to_pr3": matches,
                      "shared_model_checks": list(KINDS) if args.shared_models else []}))


if __name__ == "__main__":
    main()
