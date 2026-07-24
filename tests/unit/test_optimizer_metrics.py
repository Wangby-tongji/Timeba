import unittest

import numpy as np

from timeba.engine.optimizer import PiecewiseConstantLR
from timeba.evaluation.metrics import displacement_metrics


class OptimizerAndMetricsTest(unittest.TestCase):
    def test_historical_lr_boundaries(self):
        schedule = PiecewiseConstantLR(
            learning_rates=(1e-3, 1e-4, 4e-5),
            milestones=(32, 42),
        )
        self.assertEqual(schedule(0), 1e-3)
        self.assertEqual(schedule(31.999), 1e-3)
        self.assertEqual(schedule(32), 1e-4)
        self.assertEqual(schedule(41.999), 1e-4)
        self.assertEqual(schedule(42), 4e-5)
        self.assertEqual(schedule(52), 4e-5)

    def test_k_one_and_k_six_semantics(self):
        predictions = np.zeros((1, 6, 50, 2), dtype=np.float32)
        predictions[:, 0, :, 0] = 3.0
        predictions[:, 5, :, 0] = 1.0
        ground_truth = np.zeros((1, 50, 2), dtype=np.float32)
        valid = np.ones((1, 50), dtype=bool)

        k1 = displacement_metrics(
            predictions,
            ground_truth,
            valid,
            horizon=50,
            k=1,
        )
        k6 = displacement_metrics(
            predictions,
            ground_truth,
            valid,
            horizon=50,
            k=6,
        )
        self.assertAlmostEqual(k1["minFDE"], 3.0)
        self.assertAlmostEqual(k6["minFDE"], 0.0)
        self.assertEqual(int(k1["best_mode"][0]), 0)
        self.assertEqual(int(k6["best_mode"][0]), 1)


if __name__ == "__main__":
    unittest.main()
