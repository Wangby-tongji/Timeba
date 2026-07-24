"""Golden parity checks for the certified historical data transformations."""

import unittest
from pathlib import Path

import numpy as np

from tests.synthetic_data import make_synthetic_frame
from timeba.config.presets import EXID_PAPER, HIGHD_PAPER, NGSIM_PAPER
from timeba.data.registry import build_dataset


FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "fixtures" / "data_transformations"
)
SOURCE_COMMIT = "bd71f945d84484b2e0ebbc3988c6eb211d7db574"


class HistoricalDataTransformationParityTest(unittest.TestCase):
    CASES = (
        (
            NGSIM_PAPER,
            ("x", "y", "v_Vel", "v_Acc", "__presence__"),
        ),
        (
            HIGHD_PAPER,
            (
                "x",
                "y",
                "width",
                "height",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "precedingXVelocity",
                "__presence__",
            ),
        ),
        (
            EXID_PAPER,
            (
                "x",
                "y",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "__presence__",
            ),
        ),
    )

    def test_canonical_transformations_match_historical_golden_fixtures(self):
        array_keys = (
            "feats",
            "past",
            "ctrs",
            "gt_preds",
            "has_preds",
            "orig",
            "rot",
        )
        for experiment, expected_channels in self.CASES:
            with self.subTest(experiment=experiment.name):
                fixture_path = FIXTURE_DIRECTORY / f"{experiment.name}.npz"
                with np.load(fixture_path, allow_pickle=False) as golden:
                    canonical_dataset = build_dataset(
                        [make_synthetic_frame(experiment)],
                        experiment,
                        train=False,
                    )
                    canonical = canonical_dataset[0]

                    self.assertEqual(str(golden["source_commit"]), SOURCE_COMMIT)
                    self.assertEqual(str(golden["dataset"]), experiment.dataset)
                    self.assertEqual(int(golden["history_len"]), experiment.history_len)
                    self.assertEqual(int(golden["pred_len"]), experiment.pred_len)
                    self.assertEqual(
                        tuple(golden["channels"].tolist()),
                        expected_channels,
                    )
                    self.assertEqual(
                        tuple(golden["actor_order"].tolist()),
                        ("AGENT", "OTHERS"),
                    )
                    self.assertEqual(
                        canonical_dataset.resolved_features.tensor_channels,
                        expected_channels,
                    )
                    self.assertEqual(
                        canonical["feats"].shape,
                        (
                            len(golden["actor_order"]),
                            experiment.history_len,
                            len(expected_channels),
                        ),
                    )
                    self.assertEqual(
                        canonical["gt_preds"].shape,
                        (
                            len(golden["actor_order"]),
                            experiment.pred_len,
                            2,
                        ),
                    )
                    self.assertEqual(
                        canonical["has_preds"].shape,
                        (
                            len(golden["actor_order"]),
                            experiment.pred_len,
                        ),
                    )

                    for key in array_keys:
                        with self.subTest(experiment=experiment.name, key=key):
                            if key == "has_preds":
                                np.testing.assert_array_equal(
                                    canonical[key],
                                    golden[key],
                                )
                            else:
                                np.testing.assert_allclose(
                                    canonical[key],
                                    golden[key],
                                    rtol=0.0,
                                    atol=1e-6,
                                )
                    self.assertAlmostEqual(
                        canonical["theta"],
                        float(golden["theta"]),
                        places=7,
                    )
                    self.assertEqual(canonical["idx"], int(golden["idx"]))


if __name__ == "__main__":
    unittest.main()
