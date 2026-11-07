"""Check the post-run verifier using only synthetic URL strings and model scores."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from evaluation import gate
from tools import verify_gate_archive as check


class GateArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        reports = self.root / "gate"
        reports.mkdir()
        evidence = self.root / "evidence"
        evidence.mkdir()
        rows = [
            dict(input_id=str(i), raw_url=f"https://example.org/{i}", y_mal=i % 2)
            for i in range(4)
        ]
        self.partition = dict(rows=4, malicious=2)

        def t1(features):
            return [0.01] * len(features)

        def t2(urls, features):
            return [0.1] * len(urls)

        self.policy = gate.select(rows, t1, t2, reports, "models", "gate-file")
        self.policy_sha = gate.sha256(reports / "policy.json")
        gate.write_json(
            evidence / "experiment-b-selection.json",
            dict(model_manifest_sha256="models", split_manifest_sha256=gate.SPLIT_HASH),
        )
        gate.write_json(
            evidence / "b-split-manifest.json",
            dict(
                files={"gate_selection.jsonl": "gate-file"},
                summary={"gate_selection": self.partition},
            ),
        )
        self.candidates = gate.read_json(reports / "gate-candidates.json")
        self.timings = gate.read_json(reports / "gate-latencies.json")
        self.archive = self.root / "gate.zip"
        with zipfile.ZipFile(self.archive, "x") as bundle:
            bundle.writestr("evaluation/policy.sha256", self.policy_sha + "\n")
            for p in reports.iterdir():
                bundle.write(p, "evaluation/gate/" + p.name)

    def test_archive_round_trip_and_wrong_log_digests(self):
        archive_sha = gate.sha256(self.archive)
        with patch.object(check, "ROOT", self.root):
            result = check.verify_archive(self.archive, archive_sha, self.policy_sha)
            self.assertEqual(result["tau"], self.policy["tau"])
            self.assertEqual(len(result["candidates"]), 8)
            self.assertEqual(
                result["candidates"][0]["t2_calls"], self.partition["rows"]
            )
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                check.verify_archive(self.archive, "0" * 64, self.policy_sha)
            with self.assertRaisesRegex(ValueError, "policy digest"):
                check.verify_archive(self.archive, archive_sha, "0" * 64)

    def test_rejects_misses_eligibility_and_policy_mismatches(self):
        for field, wrong in [
            ("additional_fn", 1),
            ("allowed", False),
            ("call_rate", 0.5),
        ]:
            with self.subTest(field=field):
                candidates = copy.deepcopy(self.candidates)
                candidates[0][field] = wrong
                with self.assertRaises(ValueError):
                    check.check_candidates(
                        candidates, self.policy, self.timings, self.partition
                    )
        with self.assertRaisesRegex(ValueError, "Frozen policy disagrees"):
            check.check_candidates(
                self.candidates,
                dict(self.policy, tau=0.2),
                self.timings,
                self.partition,
            )

    def test_rejects_missing_duplicate_or_corrupt_measurement(self):
        for fault in ("missing", "duplicate", "raw"):
            with self.subTest(fault=fault):
                timings = copy.deepcopy(self.timings)
                ms = timings["measurements"]
                if fault == "missing":
                    ms.pop()
                elif fault == "duplicate":
                    ms[-1] = ms[0]
                else:
                    ms[-1]["raw_seconds"][0] += 1
                with self.assertRaises(ValueError):
                    check.check_candidates(
                        self.candidates, self.policy, timings, self.partition
                    )
