"""Compare canonical transformations with the actual historical loader code."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from tests.synthetic_data import make_synthetic_frame
from timeba.config.presets import EXID_PAPER, HIGHD_PAPER, NGSIM_PAPER
from timeba.data.registry import build_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _SyntheticSequence:
    def __init__(self, frame, city):
        self.seq_df = frame
        self.city = city


class _SyntheticLoader:
    def __init__(self, frame, city):
        self.sequence = _SyntheticSequence(frame, city)

    def __getitem__(self, _index):
        return self.sequence


def _load_historical_module(relative_path, module_name):
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_item(relative_path, module_name, frame):
    module = _load_historical_module(relative_path, module_name)
    dataset = module.ArgoDataset.__new__(module.ArgoDataset)
    dataset.config = {
        "rot_aug": False,
        "pred_range": [-100.0, 100.0, -100.0, 100.0],
    }
    dataset.train = False
    dataset.avl = _SyntheticLoader(frame, city="synthetic")
    data = dataset.read_argo_data(0)
    data = dataset.get_obj_feats(data)
    data["idx"] = 0
    return data


class HistoricalDataTransformationParityTest(unittest.TestCase):
    CASES = (
        (
            NGSIM_PAPER,
            "visualize_dev/data.py",
            "timeba_historical_ngsim_24_50",
            ("x", "y", "v_Vel", "v_Acc", "__presence__"),
        ),
        (
            HIGHD_PAPER,
            "datah.py",
            "timeba_historical_highd",
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
            "datae.py",
            "timeba_historical_exid",
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

    def test_historical_and_canonical_transformations_match(self):
        array_keys = (
            "feats",
            "past",
            "ctrs",
            "gt_preds",
            "has_preds",
            "orig",
            "rot",
        )
        for experiment, path, module_name, expected_channels in self.CASES:
            with self.subTest(experiment=experiment.name):
                frame = make_synthetic_frame(experiment)
                historical = _historical_item(
                    path,
                    module_name,
                    frame,
                )
                canonical_dataset = build_dataset(
                    [frame],
                    experiment,
                    train=False,
                )
                canonical = canonical_dataset[0]

                self.assertEqual(
                    canonical_dataset.resolved_features.tensor_channels,
                    expected_channels,
                )
                self.assertEqual(
                    canonical["feats"].shape[1:],
                    (
                        experiment.history_len,
                        len(expected_channels),
                    ),
                )
                self.assertEqual(
                    canonical["gt_preds"].shape[1:],
                    (experiment.pred_len, 2),
                )
                self.assertEqual(
                    canonical["has_preds"].shape[1:],
                    (experiment.pred_len,),
                )

                for key in array_keys:
                    with self.subTest(experiment=experiment.name, key=key):
                        if key == "has_preds":
                            np.testing.assert_array_equal(
                                canonical[key],
                                historical[key],
                            )
                        else:
                            np.testing.assert_allclose(
                                canonical[key],
                                historical[key],
                                rtol=0.0,
                                atol=1e-6,
                            )
                self.assertAlmostEqual(
                    canonical["theta"],
                    historical["theta"],
                    places=7,
                )
                self.assertEqual(canonical["idx"], historical["idx"])


if __name__ == "__main__":
    unittest.main()
