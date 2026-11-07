"""Exercise the saved policy, model archive handling and final evaluation entry paths."""

import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from importlib.util import find_spec

from evaluation import build_notebook, final, gate, kaggle_run


class FinalRunTests(unittest.TestCase):
    def test_notebook_build_is_repeatable_and_cells_compile(self):
        args = ("team/model-b", 1, "a" * 64, "team/model-a")
        notebook = build_notebook.build(*args)
        self.assertEqual(notebook, build_notebook.build(*args))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "evaluation-notebook", "exec")
        with self.assertRaises(ValueError):
            build_notebook.build("team/model-b", 1, "unknown", "team/model-a")

    def test_final_notebook_requires_frozen_gate(self):
        args = ("team/model-b", 1, "a" * 64, "team/model-a")
        with self.assertRaisesRegex(ValueError, "verified gate"):
            build_notebook.build(*args, stage="final")
        result = build_notebook.build(
            *args,
            stage="final",
            gate_notebook="team/gate",
            gate_version=2,
            gate_digest="b" * 64,
        )
        self.assertIn("main('final'", "".join(result["cells"][-1]["source"]))

    def test_zip_member_must_be_unique(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            for prefix in ("a", "b"):
                z.writestr(prefix + "/experiment-b/t2.h5", b"data")
        with zipfile.ZipFile(buffer) as z, tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                kaggle_run.extract_member(z, "experiment-b/t2.h5", Path(temp) / "t2.h5")

    def test_policy_rejects_different_inference_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.json"
            spec = dict(
                experiment="B",
                stage="gate_selection",
                split_sha256=gate.SPLIT_HASH,
                tau=0.01,
                code_sha256=gate.policy_code(),
            )
            gate.write_json(p, spec)
            self.assertEqual(gate.load_policy(p, gate.sha256(p)), spec)
            spec["code_sha256"]["evaluation/gate.py"] = "0" * 64
            gate.write_json(p, spec)
            with self.assertRaisesRegex(ValueError, "policy or inference code"):
                gate.load_policy(p, gate.sha256(p))

    @unittest.skipUnless(find_spec("numpy"), "paired bootstrap requires NumPy")
    def test_four_routes_and_all_malicious_holdout(self):
        urls = ["http://example.org/a", "//example.org/path/https://example.net"]
        records = [
            dict(
                input_id=str(i), raw_url=u, y_mal=1, group_id=str(i), source=["example"]
            )
            for i, u in enumerate(urls)
        ]

        def t1(batch):
            return [0.0 for _ in batch]

        def t2(urls, features):
            return [0.1 for _ in urls]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "final"
            report = final.evaluate(records, t1, t2, 0.001, [10, 20, 30], out)
            self.assertEqual(report["comparison"]["additional_fn"], 2)
            self.assertEqual(report["comparison"]["recall_drop"], 1)
            self.assertEqual(report["bootstrap"]["valid"], 2000)
            self.assertEqual(report["bootstrap"]["intervals"]["recall_drop"], [1, 1])
            self.assertIsNone(report["metrics"]["t2"]["fpr"])
            self.assertIsNone(report["metrics"]["t2"]["balanced_accuracy"])
            self.assertEqual(report["subgroups"]["has_scheme=False"]["rows"], 1)
            self.assertFalse(report["goals"]["recall_loss_at_most_one_point"])
            preds = [
                json.loads(line)
                for line in (out / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(preds), 8)
            self.assertFalse(
                any(r["t2_called"] for r in preds if r["mode"] == "cascade")
            )
            self.assertEqual(
                len(
                    json.loads((out / "batch-timings.json").read_text())["measurements"]
                ),
                20,
            )
