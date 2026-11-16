"""Shared validation and execution-source provenance must fail closed."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "clean_rerun_common",
    Path(__file__).resolve().parents[1] / "experiments/clean_rerun/common.py",
)
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)

try:
    import numpy as np
except ImportError:
    np = None


class ExecutionSourceTests(unittest.TestCase):
    def test_snapshot_preserves_crlf_and_rejects_modified_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = "# original execution\r\nvalue = 1\r\n"
            digest = hashlib.sha256(content.encode()).hexdigest()
            record = {"experiment.py": {"sha256": digest, "utf8": content}}
            (root / "source-snapshot.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with patch.multiple(
                common, ROOT=root, CODE_DIR=root / "missing", SOURCE=root / "missing"
            ):
                common.verify_execution_source("experiment.py", digest)
                record["experiment.py"]["utf8"] = content.replace("1", "2")
                (root / "source-snapshot.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "unverified_execution_source"
                ):
                    common.verify_execution_source("experiment.py", digest)

    def test_snapshot_does_not_accept_a_false_declared_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "0" * 64
            (root / "source-snapshot.json").write_text(
                json.dumps({"experiment.py": {"sha256": digest, "utf8": "different"}}),
                encoding="utf-8",
            )
            with (
                patch.multiple(common, ROOT=root, CODE_DIR=root / "missing"),
                self.assertRaisesRegex(RuntimeError, "unverified_execution_source"),
            ):
                common.verify_execution_source("experiment.py", digest)

    def test_second_cohort_cannot_silently_change_execution_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, source, run = root / "code", root / "source", root / "run"
            code.mkdir()
            run.mkdir()
            for name in (
                "common.py",
                "prepare.py",
                "refine_inputs.py",
                "finalize_data.py",
                "experiment.py",
                "learning_curves.py",
            ):
                (code / name).write_text("# frozen\n", encoding="utf-8")
            for name in common.REFERENCE_FILES:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# reference\n", encoding="utf-8")
            with patch.multiple(common, ROOT=run, CODE_DIR=code, SOURCE=source):
                common.snapshot_sources()
                before = (run / "source-snapshot.json").read_bytes()
                common.snapshot_sources()
                (code / "experiment.py").write_text("# modified\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, "execution_source_changed_between_cohorts"
                ):
                    common.snapshot_sources()
                self.assertEqual(before, (run / "source-snapshot.json").read_bytes())

    def test_evaluation_snapshot_keeps_the_training_snapshot_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = "# old training\r\n"
            new = "# corrected evaluation\r\n"
            for name, content in (
                ("source-snapshot.json", old),
                ("evaluation-source-snapshot.json", new),
            ):
                digest = hashlib.sha256(content.encode()).hexdigest()
                (root / name).write_text(
                    json.dumps({"experiment.py": {"sha256": digest, "utf8": content}}),
                    encoding="utf-8",
                )
            before = (root / "source-snapshot.json").read_bytes()
            with patch.multiple(common, ROOT=root, CODE_DIR=root / "missing"):
                for content in (old, new):
                    common.verify_execution_source(
                        "experiment.py", hashlib.sha256(content.encode()).hexdigest()
                    )
            self.assertEqual(before, (root / "source-snapshot.json").read_bytes())


class TrainingArtifactTests(unittest.TestCase):
    def test_intact_seal_and_same_code_weight_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in common.TRAINING_ARTIFACTS:
                (root / name).write_bytes(b"original artifact")
            common.seal_training_artifacts(root)
            common.verify_training_artifacts(root)
            (root / "t2.h5").write_bytes(b"changed weights")
            with self.assertRaisesRegex(
                RuntimeError, "training_artifact_changed:t2.h5"
            ):
                common.verify_training_artifacts(root)
            with self.assertRaisesRegex(
                RuntimeError, "training_manifest_already_exists"
            ):
                common.seal_training_artifacts(root)

    def test_missing_seal_and_omitted_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "missing_training_manifest"):
                common.verify_training_artifacts(root)
            (root / "training-manifest.json").write_text(
                json.dumps({"status": "PASS", "files": {}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "invalid_training_manifest"):
                common.verify_training_artifacts(root)


@unittest.skipIf(
    np is None, "NumPy is optional in the repository's stdlib-only CI suite"
)
class SharedValidationTests(unittest.TestCase):
    def parts(self):
        return {
            "train": np.array([0, 1]),
            "model": np.array([4]),
            "calibration": np.array([2]),
            "policy": np.array([3]),
            "test": np.array([5]),
        }

    def test_all_former_validation_rows_are_used_once(self):
        self.assertEqual(common.validation_indices(self.parts()).tolist(), [2, 3, 4])

    def test_test_rows_are_rejected(self):
        parts = self.parts()
        parts["policy"] = np.array([5])
        with self.assertRaisesRegex(
            RuntimeError, "invalid_shared_validation_partition"
        ):
            common.validation_indices(parts)

    def test_training_rows_and_duplicate_rows_are_rejected(self):
        for bad in (0, 2):
            with self.subTest(bad=bad):
                parts = self.parts()
                parts["policy"] = np.array([bad])
                with self.assertRaisesRegex(
                    RuntimeError, "invalid_shared_validation_partition"
                ):
                    common.validation_indices(parts)


if __name__ == "__main__":
    unittest.main()
