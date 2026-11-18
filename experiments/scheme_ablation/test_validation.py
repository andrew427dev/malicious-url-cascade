"""Regression checks for the defensive scheme experiment (use the #3 ML environment)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import canonical
import verify_inputs as vi
from common import TRAINING_ARTIFACTS, seal_training_artifacts, sha, write_json


class InputTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data-final"
        self.data.mkdir()

    def test_ordered_ids_labels_and_index_membership(self):
        roles = ["train", "test", "train", "policy", "model", "calibration"]
        df = pd.DataFrame({"id": [f"{i:064x}" for i in range(6)], "y_mal": [0, 1] * 3,
                           "random_split": roles, "domain_split": roles[::-1]})
        for kind in vi.KINDS:
            parts = {p: np.flatnonzero(df[kind + "_split"] == p) for p in vi.ROLES}
            np.savez(self.data / (kind + "_indices.npz"), **parts)
        before = vi.ordered_cohorts(self.data, df)
        for col in ("id", "y_mal"):
            changed = df.copy()
            if col == "id":
                changed.loc[[0, 2], col] = list(reversed(df.loc[[0, 2], col].tolist()))
            else:
                changed.loc[0, col] = 1
            self.assertNotEqual(before["random"]["train"], vi.ordered_cohorts(self.data, changed)["random"]["train"])
        parts["test"] = parts["train"]
        np.savez(self.data / "domain_indices.npz", **parts)
        with self.assertRaisesRegex(ValueError, "split_index_mismatch:domain:test"):
            vi.ordered_cohorts(self.data, df)

    def test_policy_and_data_are_bound_to_the_training_run(self):
        out = self.root / "random"
        reference = self.root / "reference" / "random"
        out.mkdir()
        reference.mkdir(parents=True)
        for name in TRAINING_ARTIFACTS:
            (out / name).write_bytes(name.encode("ascii"))
        write_json(self.data / "manifest.json", {"files": {}})
        write_json(out / "settings.json", {"kind": "random", "data_manifest_sha256": sha(self.data / "manifest.json")})
        seal_training_artifacts(out)
        policy = {"frozen_before_test": True, "model_hashes": {
            n: sha(out / n) for n in ("t2.h5", "vocabulary.json", "logistic.joblib", "boost.joblib")}}
        write_json(out / "policy-frozen.json", policy)
        complete = {"status": "PASS", "kind": "random", "files": {p.name: sha(p) for p in out.iterdir()}}
        for target in (out, reference):
            write_json(target / "complete.json", complete)
        with patch.object(vi, "BASELINE", reference.parent):
            vi.verify_shared_run(self.root, "random")
            write_json(out / "policy-frozen.json", {**policy, "frozen_before_test": False})
            with self.assertRaisesRegex(ValueError, "completed_artifact_changed:policy-frozen"):
                vi.verify_shared_run(self.root, "random")
            write_json(out / "policy-frozen.json", policy)
            write_json(self.data / "manifest.json", {"files": {"changed": True}})
            with self.assertRaisesRegex(ValueError, "training_data_mismatch"):
                vi.verify_shared_run(self.root, "random")

    def test_canonical_manifest_counts_retained_rows_and_roles(self):
        urls = ["http://example.test/a", "example.test/a", "example.test/b", "example.test/c"]
        df = pd.DataFrame({"url_key": urls, "domain": ["example.test"] * 4, "y_mal": [0, 1, 0, 1],
                           "random_split": ["train", "test", "train", "test"],
                           "domain_split": ["train", "train", "train", "train"]})
        for kind in vi.KINDS:
            np.save(self.data / (kind + "-encoded.npy"), np.array([[1], [1], [2], [3]]))
            np.savez(self.data / (kind + "_indices.npz"), **{
                p: np.flatnonzero(df[kind + "_split"] == p) for p in vi.ROLES})
        names = [p.name for p in self.data.iterdir()] + ["clean.csv", "features.npy"]
        write_json(self.data / "manifest.json", {"files": dict.fromkeys(names), "stages": [], "final": {"rows": 4}})
        dest = self.root / "out"
        dest.mkdir()
        canonical.build(self.data, dest, df)
        meta = vi.read(dest / "manifest.json")
        self.assertEqual(meta["final"], {"rows": 2, "benign": 1, "malicious": 1})
        self.assertEqual(meta["splits"]["random"]["test"], {
            "rows": 1, "benign": 0, "malicious": 1, "domains": 1, "fraction": 0.5})
        self.assertEqual(meta["splits"]["domain"]["test"]["rows"], 0)
        self.assertEqual(meta["canonical_collision_exclusion"]["rows"], 2)
        for name, digest in meta["files"].items():
            self.assertEqual(sha(dest / name), digest)
        saved = pd.read_csv(dest / "clean.csv")
        self.assertEqual(saved.url_key.tolist(), urls[2:])
        for kind in vi.KINDS:
            vi.checked_parts(dest, saved, kind)
            self.assertEqual(len(np.load(dest / (kind + "-encoded.npy"))), len(saved))


if __name__ == "__main__":
    unittest.main()
