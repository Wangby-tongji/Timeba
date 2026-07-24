import gc
import os
import tempfile
import unittest
from pathlib import Path

import torch

from tests.synthetic_data import make_synthetic_frame
from timeba.config.presets import EXID_PAPER, HIGHD_PAPER, NGSIM_PAPER
from timeba.data.collate import collate_trajectory_batch
from timeba.data.registry import build_dataset
from timeba.engine.checkpoint import load_checkpoint, save_checkpoint
from timeba.engine.evaluator import evaluate_batch
from timeba.engine.factory import build_pipeline
from timeba.engine.trainer import train_step


RUN_CUDA_INTEGRATION = (
    os.environ.get("TIMEBA_RUN_CUDA_INTEGRATION") == "1"
    and torch.cuda.is_available()
)


@unittest.skipUnless(
    RUN_CUDA_INTEGRATION,
    "set TIMEBA_RUN_CUDA_INTEGRATION=1 on a CUDA runtime",
)
class SyntheticPipelineIntegrationTest(unittest.TestCase):
    def test_forward_loss_backward_optimizer_checkpoint_and_evaluation(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        expected_input_dims = {
            "ngsim_paper": 5,
            "highd_paper": 10,
            "exid_paper": 7,
        }
        for experiment in (NGSIM_PAPER, HIGHD_PAPER, EXID_PAPER):
            with self.subTest(experiment=experiment.name):
                frame = make_synthetic_frame(experiment)
                dataset = build_dataset([frame], experiment, train=False)
                batch = collate_trajectory_batch([dataset[0]])
                components = build_pipeline(experiment, device="cuda")
                self.assertEqual(
                    components.features.input_dim,
                    expected_input_dims[experiment.name],
                )

                step = train_step(components, batch, epoch=0.5)
                self.assertTrue(torch.isfinite(step["loss"]).item())
                self.assertGreater(step["num_reg"], 0)
                self.assertEqual(step["learning_rate"], 1e-3)

                with tempfile.TemporaryDirectory() as directory:
                    checkpoint_path = Path(directory) / "epoch_001.ckpt"
                    save_checkpoint(
                        checkpoint_path,
                        components,
                        epoch=1.0,
                        extra={"synthetic_only": True},
                    )
                    self.assertTrue(checkpoint_path.is_file())

                    reloaded = build_pipeline(experiment, device="cuda")
                    payload, incompatible = load_checkpoint(
                        checkpoint_path,
                        reloaded.model,
                        optimizer=reloaded.optimizer,
                        strict=True,
                        map_location="cpu",
                    )
                    self.assertEqual(payload["epoch"], 1.0)
                    self.assertEqual(payload["extra"]["synthetic_only"], True)
                    self.assertEqual(incompatible.missing_keys, [])
                    self.assertEqual(incompatible.unexpected_keys, [])

                    evaluated = evaluate_batch(reloaded, batch)
                    self.assertEqual(
                        tuple(evaluated["cls"].shape),
                        (1, experiment.num_modes),
                    )
                    self.assertEqual(
                        tuple(evaluated["reg"].shape),
                        (
                            1,
                            experiment.num_modes,
                            experiment.pred_len,
                            2,
                        ),
                    )
                    self.assertIn("K=1", evaluated["metrics"])
                    self.assertIn(
                        f"K={experiment.num_modes}",
                        evaluated["metrics"],
                    )

                del reloaded
                del components
                del batch
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
