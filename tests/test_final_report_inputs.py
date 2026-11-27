"""Exercise partial archives, broken seals and read-only operation without scientific packages."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

V2CODE = Path(__file__).resolve().parents[1] / "experiments/final_report_v2"
with mock.patch.object(sys, "path", [str(V2CODE), *sys.path]):
    spec = importlib.util.spec_from_file_location("input_preflight", V2CODE / "check_inputs.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)


def put(path, data=b"opaque input; never deserialize\r\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_json(path, data):
    put(path, json.dumps(data).encode("utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}


class InputPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.v1 = self.root / "final-v1"
        self.v2 = self.root / "final-v2"
        self.code = self.v1 / "code"

    def fixture(self):
        # Only file-level evidence is valid. The binary placeholders cannot be loaded as models,
        # NumPy arrays or pandas pickles, so the passing case also checks the limited scope.
        for path in preflight.required_files(self.v1, self.code):
            put(path, b"{}" if path.suffix == ".json" else b"opaque fixture\r\n")
        write_json(self.v1 / "results/verification.json", {"status": "PASS"})
        utils = {"Utils/" + name: sha(self.v1 / "source/Utils" / name)
                 for name in ("DataProcessing.py", "FeatureCreation.py", "Model.py")}
        datasets = {"dataset/" + name: {"sha256": sha(self.v1 / "source/dataset" / name)}
                    for name in preflight.DATASETS}
        write_json(self.code / "configs/upstream.json", {"source_sha256": utils, "files": datasets})
        for kind in preflight.KINDS:
            names = ("length", "logistic", "boost", "logistic-omit") if kind in preflight.TEAM else ("length",)
            hashes = {name: sha(put(self.v1 / kind / "stage1" / (name + ".joblib"))) for name in names}
            write_json(self.v1 / kind / "stage1/t1-selected.json", {"hashes": hashes})
        for kind in preflight.TEAM:
            out = self.v1 / kind
            write_json(out / "vocab.json", ["", "[UNK]", "example"])
            record = {
                "vocab.json": out / "vocab.json",
                "runner/train_gru.py": put(self.code / "experiments/upstream_partition/train_gru.py"),
                "source/Utils/Model.py": self.v1 / "source/Utils/Model.py",
                "team/urlcascade/features.py": self.code / "urlcascade/features.py",
            }
            for role in ("train", "validation", "test"):
                for suffix in ("tokens.npy", "features.npy", "rows.csv.gz"):
                    name = role + "-" + suffix
                    path = out / name
                    if not path.exists():
                        put(path)
                    record[name] = path
            # Windows-created manifests use backslashes in some of their relative keys.
            write_json(out / "training-inputs.json", {
                "capture": "before_fit", "files": {key.replace("/", "\\"): sha(path) for key, path in record.items()}})
            write_json(out / "trained.json", {"weights_sha256": sha(out / "gru.h5"),
                                                "vocabulary_sha256": sha(out / "vocab.json")})

    def run_check(self, **kwargs):
        return preflight.check(self.v1, self.v2, self.code, **kwargs)

    def test_complete_static_fixture_passes_without_output_or_deserialization(self):
        self.fixture()
        # Use the supported original layout and disable site packages entirely.
        for name in ("check_inputs.py", "v2paths.py"):
            put(self.v2 / "code" / name, (V2CODE / name).read_bytes())
        before = snapshot(self.root)
        env = {k: v for k, v in os.environ.items() if not k.startswith("URLCASCADE_")}
        env["PYTHONUTF8"] = "1"
        result = subprocess.run([sys.executable, "-B", "-S", str(self.v2 / "code/check_inputs.py")],
                                env=env, capture_output=True, text=True, encoding="utf8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["packages_checked"])
        self.assertGreater(report["recorded_hashes_checked"], 0)
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.v2 / "results").exists())

    def test_partial_archive_lists_known_missing_inputs_together(self):
        self.fixture()
        names = ("team-random-original/train.pkl", "domain-original/validation.pkl",
                 "team-random-original/controls-final.json", "team-domain-original/fixed-prefix.json",
                 "results/sensitivity.csv", "team-domain-original/stage1/logistic-omit.joblib")
        for name in names:
            (self.v1 / name).unlink()
        before = snapshot(self.root)
        report = self.run_check()
        missing = {i["path"] for i in report["issues"] if i["kind"] == "missing_file"}
        self.assertEqual(missing, {str(self.v1 / name) for name in names})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.v2.exists())

    def test_crlf_to_lf_change_is_reported_and_never_resealed(self):
        self.fixture()
        path = self.code / "experiments/upstream_partition/train_gru.py"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        before = snapshot(self.root)
        report = self.run_check()
        mismatches = [i for i in report["issues"] if i["kind"] == "sha256_mismatch"]
        self.assertEqual({i["path"] for i in mismatches}, {str(path)})
        self.assertTrue(all("CRLF/LF" in i["detail"] for i in mismatches))
        self.assertEqual(snapshot(self.root), before)

    def test_missing_seal_key_and_changed_model_fail_before_inference(self):
        self.fixture()
        record = self.v1 / "team-random-original/training-inputs.json"
        seal = json.loads(record.read_text())
        del seal["files"]["train-tokens.npy"]
        write_json(record, seal)
        weights = self.v1 / "team-domain-original/gru.h5"
        weights.write_bytes(b"different checkpoint")
        report = self.run_check()
        self.assertTrue(any(i["kind"] == "invalid_record" and "train-tokens.npy" in i["detail"] for i in report["issues"]))
        self.assertTrue(any(i["kind"] == "sha256_mismatch" and i["path"] == str(weights) for i in report["issues"]))

    def test_null_list_empty_and_failed_records_cannot_pass(self):
        self.fixture()
        cases = (
            ("results/verification.json", None), ("results/verification.json", []),
            ("results/verification.json", {"status": "FAIL"}),
            ("team-random-original/trained.json", None), ("team-random-original/trained.json", {}),
            ("team-random-original/training-inputs.json", None),
            ("team-random-original/training-inputs.json", {"capture": "before_fit", "files": {}}),
            ("team-random-original/stage1/t1-selected.json", None),
            ("team-random-original/stage1/t1-selected.json", {"hashes": {}}),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                path = self.v1 / name
                previous = path.read_bytes()
                write_json(path, value)
                report = self.run_check()
                self.assertEqual(report["status"], "FAIL")
                self.assertTrue(any(i["kind"] == "invalid_record" and i["path"] == str(path) for i in report["issues"]))
                path.write_bytes(previous)

    def test_package_discovery_is_optional_and_does_not_import_packages(self):
        self.fixture()
        with mock.patch.object(preflight.importlib.util, "find_spec", side_effect=lambda name: None if name == "tensorflow" else object()) as find:
            self.assertEqual(self.run_check()["status"], "PASS")
            find.assert_not_called()
            report = self.run_check(check_packages=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([i["path"] for i in report["issues"] if i["kind"] == "missing_package"], ["tensorflow"])
        self.assertEqual(find.call_count, len(preflight.PACKAGES))

    def test_existing_v1_output_protection_is_preserved(self):
        self.fixture()
        before = snapshot(self.root)
        report = preflight.check(self.v1, self.v1 / "new-output", self.code)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["issues"][0]["kind"], "protected_output")
        self.assertIn("refused[v1_protected]", report["issues"][0]["detail"])
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.v1 / "new-output").exists())


if __name__ == "__main__":
    unittest.main()
