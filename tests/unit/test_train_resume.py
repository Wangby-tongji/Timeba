import json
import tempfile
import unittest
from pathlib import Path

from scripts.common import load_metrics_history, write_json


class TrainResumeMetricsTest(unittest.TestCase):
    def test_resume_preserves_and_appends_existing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            original = [
                {"epoch": 1, "train_loss": 3.0},
                {"epoch": 2, "train_loss": 2.0},
            ]
            write_json(path, original)

            resumed = load_metrics_history(path, resume_epoch=2)
            resumed.append({"epoch": 3, "train_loss": 1.0})
            write_json(path, resumed)

            self.assertEqual(json.loads(path.read_text()), original + [resumed[-1]])

    def test_resume_rejects_metrics_ahead_of_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            write_json(path, [{"epoch": 3, "train_loss": 1.0}])
            with self.assertRaisesRegex(ValueError, "ahead"):
                load_metrics_history(path, resume_epoch=2)

    def test_resume_rejects_malformed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text('{"epoch": 1}\n')
            with self.assertRaisesRegex(ValueError, "JSON list"):
                load_metrics_history(path, resume_epoch=1)


if __name__ == "__main__":
    unittest.main()
