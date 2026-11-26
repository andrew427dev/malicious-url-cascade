"""Check the sealed-input record of the upstream flow and the v1 protection of the v2 scripts.

Importing this module needs only the standard library. The two tests that open real PDFs need PyMuPDF
and are skipped without it. Every scratch folder is a TemporaryDirectory; nothing is written to the
repository or to the v1/v2 archive folders.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "experiments" / "upstream_partition"
V2CODE = REPO / "experiments" / "final_report_v2"
MARKER = "source-manifest.json"

ROLE_FILES = tuple(
    f"{role}-{kind}"
    for role in ("train", "validation", "test")
    for kind in ("tokens.npy", "features.npy", "rows.csv.gz")
)
DATASETS = ("df_final.csv", "all_top_domains_merged.csv", "df_another_validation.csv")
UTILS = ("DataProcessing.py", "FeatureCreation.py", "Model.py")

# What the original v1 comparison models recorded right before training (no assignment table).
EXPECTED_SEAL_KEYS = (
    {f"source/dataset/{n}" for n in DATASETS}
    | {f"source/Utils/{n}" for n in UTILS}
    | {"runner/inputs.py", "runner/train_gru.py", "runner/team_data.py"}
    | {
        "team/urlcascade/adapters.py",
        "team/urlcascade/features.py",
        "team/experiments/clean_rerun/prepare.py",
    }
    | {"vocab.json"}
    | set(ROLE_FILES)
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf8")


def _touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fake_v1(root):
    """A folder that looks like a finished v1 run to the v2 scripts."""
    _touch(root / MARKER, b"{}")
    _touch(root / "results" / "verification.json", b"{}")
    return root


def _run(args, cwd, v1=None, v2=None, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("URLCASCADE_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    if v1 is not None:
        env["URLCASCADE_V1_DIR"] = str(v1)
    if v2 is not None:
        env["URLCASCADE_V2_DIR"] = str(v2)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-B", *map(str, args)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf8",
        errors="replace",
        timeout=180,
    )


PRINT_FOLDERS = "import v2common as c; print(c.V1); print(c.V2); print(c.CODE)"


class SealRecordTests(unittest.TestCase):
    def test_a1_seal_without_assignment_table_matches_original_record(self):
        """A1: the run_study flow has no assignment table; its seal must still succeed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name in DATASETS:
                _touch(root / "source" / "dataset" / name, name.encode())
            for name in UTILS:
                _touch(root / "source" / "Utils" / name, name.encode())
            _touch(root / "random-original" / "vocab.json", b'{"vocab": []}')
            for name in ROLE_FILES:
                _touch(root / "random-original" / name, name.encode())

            fake = types.ModuleType("inputs")
            fake.ROOT = root
            fake.SOURCE = root / "source"
            fake.REPO = REPO
            fake.sha = _sha
            fake.write_json = _write_json
            fake.FEATURE_NAMES = ()
            with mock.patch.dict(sys.modules, {"inputs": fake}), mock.patch.object(
                sys, "dont_write_bytecode", True
            ):
                spec = importlib.util.spec_from_file_location(
                    "provenance_under_test", UPSTREAM / "provenance.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.seal("random-original")
                record = json.loads(
                    (root / "random-original" / "training-inputs.json").read_text(encoding="utf8")
                )
                keys = {k.replace("\\", "/") for k in record["files"]}
                self.assertEqual(keys, EXPECTED_SEAL_KEYS)
                self.assertEqual(len(keys), 22)
                module.verify_seal("random-original")


class V2ProtectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()

    def assert_kept(self, path, data):
        self.assertTrue(path.is_file(), f"{path} was deleted")
        self.assertEqual(path.read_bytes(), data)

    def assert_refused(self, result, code):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"refused[{code}]", result.stderr)

    def test_b1_repository_layout_requires_explicit_folders(self):
        v1 = _fake_v1(self.tmp / "v1")
        v2 = self.tmp / "v2"
        cases = (
            ("no variable", None, None, "URLCASCADE_V2_DIR"),
            ("v1 only", v1, None, "URLCASCADE_V2_DIR"),
            ("v2 only", None, v2, "URLCASCADE_V1_DIR"),
        )
        for label, set_v1, set_v2, named in cases:
            with self.subTest(label):
                result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=set_v1, v2=set_v2)
                self.assert_refused(result, "missing_env")
                self.assertIn(named, result.stderr)
                self.assertFalse(v2.exists())

    def test_b2_v1_must_hold_both_run_markers(self):
        v2 = self.tmp / "v2"
        for missing in (MARKER, "results/verification.json"):
            with self.subTest(missing=missing):
                v1 = _fake_v1(self.tmp / ("v1-without-" + missing.replace("/", "-")))
                (v1 / missing).unlink()
                result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=v1, v2=v2)
                self.assert_refused(result, "v1_not_run_folder")
                self.assertFalse(v2.exists())

    def test_b3_new_subfolder_of_v1_is_refused_before_creation(self):
        v1 = _fake_v1(self.tmp / "v1")
        result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=v1, v2=v1 / "newsub")
        self.assert_refused(result, "v1_protected")
        self.assertFalse((v1 / "newsub").exists())

    def test_b4_v1_itself_is_refused_as_v2(self):
        v1 = _fake_v1(self.tmp / "v1")
        before = sorted(p.relative_to(v1).as_posix() for p in v1.rglob("*"))
        result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=v1, v2=v1)
        self.assert_refused(result, "v1_protected")
        self.assertEqual(sorted(p.relative_to(v1).as_posix() for p in v1.rglob("*")), before)

    def test_b4b_marker_on_the_target_folder_itself_is_refused(self):
        v1 = _fake_v1(self.tmp / "v1")
        v2 = self.tmp / "v2"
        _touch(v2 / "report" / MARKER, b"{}")
        result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=v1, v2=v2)
        self.assert_refused(result, "v1_protected")
        self.assertFalse((v2 / "results").exists())

    def test_b5_evidence_copy_as_v1_does_not_unlock_the_original_run(self):
        run = _fake_v1(self.tmp / "run")
        copy = _fake_v1(run / "reviews" / "copy")
        result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=copy, v2=run / "out")
        self.assert_refused(result, "v1_protected")
        self.assertFalse((run / "out").exists())

    def test_b6_folder_resolution_in_both_layouts(self):
        v1 = _fake_v1(self.tmp / "v1")
        v2 = self.tmp / "v2"
        result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=v1, v2=v2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [Path(line) for line in result.stdout.splitlines()], [v1, v2, REPO]
        )
        self.assertTrue((v2 / "results").is_dir())

        # Importing v2paths alone has no side effect: no exit and no folder, with no variable set.
        quiet = self.tmp / "quiet"
        quiet.mkdir()
        before = sorted(p.name for p in self.tmp.iterdir())
        code_before = sorted(p.name for p in V2CODE.iterdir())
        result = _run(
            ["-c", f"import sys; sys.path.insert(0, {str(V2CODE)!r}); import v2paths"], quiet
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), before)
        self.assertEqual(list(quiet.iterdir()), [])
        self.assertEqual(sorted(p.name for p in V2CODE.iterdir()), code_before)

        # Original output layout: final-v1 next to final-v2/code, no repository around it.
        archive = self.tmp / "archive"
        code = archive / "final-v2" / "code"
        code.mkdir(parents=True)
        for name in ("v2common.py", "v2paths.py"):
            shutil.copy2(V2CODE / name, code / name)
        old_v1 = _fake_v1(archive / "final-v1")
        old_v2 = archive / "final-v2"
        other_v1 = _fake_v1(self.tmp / "other-v1")
        cases = (
            ("no variable", None, None, [old_v1, old_v2, old_v1 / "code"]),
            ("v1 only", other_v1, None, [other_v1, old_v2, other_v1 / "code"]),
            ("both", other_v1, self.tmp / "out", [other_v1, self.tmp / "out", other_v1 / "code"]),
        )
        for label, set_v1, set_v2, expected in cases:
            with self.subTest("original layout, " + label):
                result = _run(["-c", PRINT_FOLDERS], code, v1=set_v1, v2=set_v2)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([Path(line) for line in result.stdout.splitlines()], expected)
        with self.subTest("original layout, v2 only"):
            result = _run(["-c", PRINT_FOLDERS], code, v2=self.tmp / "out2")
            self.assert_refused(result, "missing_env")
            self.assertIn("URLCASCADE_V1_DIR", result.stderr)
            self.assertFalse((self.tmp / "out2").exists())

    @unittest.skipUnless(os.name == "nt", "Windows 경로 대소문자 시험")
    def test_b7_case_only_difference_is_the_same_folder_on_windows(self):
        v1 = _fake_v1(self.tmp / "v1")
        swapped = Path(str(v1).swapcase()) / "Sub"
        result = _run(["-c", PRINT_FOLDERS], V2CODE, v1=v1, v2=swapped)
        self.assert_refused(result, "v1_protected")
        self.assertFalse((v1 / "Sub").exists())

    def test_b8_render_pages_refuses_pdf_inside_v1_and_keeps_images(self):
        v1 = _fake_v1(self.tmp / "v1")
        _touch(v1 / "report" / "x.pdf", b"%PDF-1.4")
        keep = _touch(v1 / "report" / "layout-qa" / "page-01.png", b"old image")
        result = _run(["render_pages.py", v1 / "report" / "x.pdf"], V2CODE)
        self.assert_refused(result, "v1_protected")
        self.assert_kept(keep, b"old image")

    def test_b9_render_pages_refuses_a_missing_pdf_and_keeps_images(self):
        keep = _touch(self.tmp / "layout-qa" / "page-01.png", b"old image")
        result = _run(["render_pages.py", self.tmp / "missing.pdf"], V2CODE)
        self.assert_refused(result, "missing_pdf")
        self.assert_kept(keep, b"old image")

    @unittest.skipUnless(find_spec("pymupdf") is not None, "PyMuPDF 필요")
    def test_b10_unreadable_pdf_keeps_existing_images(self):
        import pymupdf

        bad = _touch(self.tmp / "bad.pdf", b"this is not a pdf")
        with self.assertRaises(Exception):
            pymupdf.open(bad)
        keep = _touch(self.tmp / "layout-qa" / "page-07.png", b"old image")
        result = _run(["render_pages.py", bad], V2CODE)
        self.assertNotEqual(result.returncode, 0)
        self.assert_kept(keep, b"old image")

    def test_b11_dryrun_output_inside_v1_is_refused(self):
        v1 = _fake_v1(self.tmp / "v1")
        v2 = self.tmp / "v2"
        script = (
            "import runpy, sys, types\n"
            "from unittest.mock import MagicMock\n"
            "needed = {'threadpoolctl': ('threadpool_limits', 'threadpool_info'),\n"
            "          'inputs': ('tf', 'np', 'pd', 'vectorizer', 'preprocess_link', 'extract_features'),\n"
            "          'evaluate': ('load_gru', 'load_stages', 'load_rows', 'route'),\n"
            "          'experiment': ('cascade_metrics',),\n"
            "          'cohorts': ('model_url',)}\n"
            "for name, attrs in needed.items():\n"
            "    module = types.ModuleType(name)\n"
            "    for attr in attrs:\n"
            "        setattr(module, attr, MagicMock())\n"
            "    sys.modules[name] = module\n"
            f"sys.argv = ['e6_rate_timing.py', 'dryrun', {str(v1 / 'new-out')!r}]\n"
            "runpy.run_path('e6_rate_timing.py', run_name='__main__')\n"
        )
        result = _run(["-c", script], V2CODE, v1=v1, v2=v2)
        self.assert_refused(result, "v1_protected")
        self.assertFalse((v1 / "new-out").exists())

    @unittest.skipUnless(find_spec("pymupdf") is not None, "PyMuPDF 필요")
    def test_b12_render_pages_replaces_old_images_in_both_layouts(self):
        import pymupdf

        archive = self.tmp / "archive"
        code = archive / "final-v2" / "code"
        code.mkdir(parents=True)
        for name in ("render_pages.py", "v2paths.py"):
            shutil.copy2(V2CODE / name, code / name)
        for label, folder in (("repository", V2CODE), ("original layout", code)):
            with self.subTest(label):
                work = self.tmp / ("work-" + label.replace(" ", "-"))
                work.mkdir()
                pdf = work / "doc.pdf"
                document = pymupdf.open()
                document.new_page().insert_text((72, 72), "page one")
                document.save(pdf)
                document.close()
                stale = _touch(work / "layout-qa" / "page-07.png", b"old image")
                result = _run(["render_pages.py", pdf], folder)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(stale.exists())
                self.assertGreater((work / "layout-qa" / "page-01.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
