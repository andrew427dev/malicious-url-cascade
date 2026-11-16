"""Use the actual evaluator for probability-boundary and pre-load integrity checks."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_ML = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(HAS_ML, "The full experiment environment is required")
class EvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = Path(__file__).resolve().parents[1] / "experiments/clean_rerun"
        with patch.object(sys, "path", [str(code), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "clean_rerun_evaluation_test", code / "experiment.py"
            )
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)

    def test_float32_half_and_both_neighbors_match_direct_decisions(self):
        import numpy as np

        half = np.float32(0.5)
        benign = np.array(
            [
                0,
                np.nextafter(half, np.float32(0)),
                half,
                np.nextafter(half, np.float32(1)),
                1,
            ],
            dtype=np.float32,
        )
        malicious = self.runner.malicious_scores(benign)
        self.assertEqual(malicious.dtype, np.float64)
        np.testing.assert_array_equal(malicious > 0.5, benign < 0.5)
        self.assertEqual(malicious[2], 0.5)

    def test_same_code_changed_weights_fail_before_model_or_dataset_loading(self):
        import common

        runner = self.runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "random"
            out.mkdir()
            for name in common.TRAINING_ARTIFACTS:
                (out / name).write_bytes(b"frozen training output")
            (out / "settings.json").write_text(
                json.dumps({"experiment_sha256": common.sha(Path(runner.__file__))}),
                encoding="utf-8",
            )
            common.seal_training_artifacts(out)
            (out / "t2.h5").write_bytes(b"tampered weights")
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner.joblib, "load") as load_model,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "training_artifact_changed:t2.h5"
                ):
                    runner.evaluate("random")
                load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
