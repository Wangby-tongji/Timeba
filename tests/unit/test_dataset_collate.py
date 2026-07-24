import unittest

import torch

from tests.synthetic_data import make_synthetic_frame
from timeba.config.presets import EXID_PAPER, HIGHD_PAPER, NGSIM_PAPER
from timeba.data.collate import collate_trajectory_batch
from timeba.data.registry import build_dataset


class DatasetAndCollateTest(unittest.TestCase):
    def test_canonical_dataset_shapes_and_batch_contract(self):
        expected_input_dims = {
            "ngsim_paper": 5,
            "highd_paper": 10,
            "exid_paper": 7,
        }
        for experiment in (NGSIM_PAPER, HIGHD_PAPER, EXID_PAPER):
            with self.subTest(experiment=experiment.name):
                dataset = build_dataset(
                    [make_synthetic_frame(experiment)],
                    experiment,
                )
                item = dataset[0]
                input_dim = expected_input_dims[experiment.name]
                self.assertEqual(
                    item["feats"].shape,
                    (2, experiment.history_len, input_dim),
                )
                self.assertEqual(
                    item["gt_preds"].shape,
                    (2, experiment.pred_len, 2),
                )
                self.assertEqual(
                    item["has_preds"].shape,
                    (2, experiment.pred_len),
                )
                self.assertTrue(item["has_preds"][0].all())
                self.assertEqual(item["ctrs"].shape, (2, 2))

                batch = collate_trajectory_batch([item])
                self.assertEqual(len(batch["feats"]), 1)
                self.assertIsInstance(batch["feats"][0], torch.Tensor)
                self.assertEqual(
                    tuple(batch["feats"][0].shape),
                    (2, experiment.history_len, input_dim),
                )
                self.assertEqual(
                    tuple(batch["gt_preds"][0].shape),
                    (2, experiment.pred_len, 2),
                )


if __name__ == "__main__":
    unittest.main()
