"""Verify the gate runner across scoring, real routing and persisted selection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import gate
from urlcascade.features import extract_features


class GateRunTests(unittest.TestCase):
    def test_selection_and_committed_outputs(self):
        urls = ["https://example.org/" + "a" * i for i in range(1, 5)]
        rows = [
            dict(input_id=str(i), raw_url=u, y_mal=i % 2) for i, u in enumerate(urls)
        ]
        features = {tuple(extract_features(u)): i for i, u in enumerate(urls)}

        def t1(items):
            return [(0.001, 0.9, 0.2, 0.01)[features[tuple(v)]] for v in items]

        def t2(items, numeric):
            return [0.1 if features[tuple(v)] % 2 else 0.5 for v in numeric]

        # Deterministic timings exercise the 5% tie and 1 percentage point loss rules.
        def bench(records, *, t1, t2, thresholds):
            self.assertEqual(thresholds, [0, 0.001, 0.005, 0.01])
            times = {0: 3, 0.001: 2, 0.005: 1.04, 0.01: 1}
            measured = [
                dict(mode="cascade", tau=tau, mean_seconds=times[tau])
                for tau in thresholds
                for _ in range(5)
            ]
            return dict(measurements=measured + [dict(mode="t2", mean_seconds=2)] * 5)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(gate, "sequential_benchmark", bench),
        ):
            out = Path(tmp)
            result = gate.select(rows, t1, t2, out, "models", "partition")
            self.assertEqual(result["tau"], 0.005)
            self.assertEqual(result["additional_fn"], 0)
            self.assertEqual(result["call_rate"], 0.75)
            self.assertEqual(json.loads((out / "policy.json").read_text()), result)
            for name, digest in result["evidence_sha256"].items():
                gate.verify_file(out / name, digest)

    def test_altered_input_file_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gate_selection.jsonl"
            p.write_text("<invalid JSON>")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                gate.load_rows(p, "0" * 64)
