"""Official config-driven Timeba evaluation CLI."""

import argparse
from pathlib import Path

import numpy as np
import torch

from timeba.engine import build_pipeline
from timeba.engine.checkpoint import load_checkpoint
from timeba.evaluation import displacement_metrics

from scripts.common import (
    add_common_arguments,
    build_split_loader,
    resolve_experiment,
    seed_everything,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = resolve_experiment(args)
    seed_everything(args.seed)
    loader = build_split_loader(
        experiment,
        args.data_root,
        args.split,
        train=False,
    )
    components = build_pipeline(experiment, device="cuda")
    load_checkpoint(
        args.checkpoint,
        components.model,
        strict=True,
        map_location="cpu",
    )
    components.model.eval()

    predictions = []
    ground_truth = []
    valid = []
    with torch.no_grad():
        for batch in loader:
            output = components.model(dict(batch))
            predictions.extend(scene[0].cpu().numpy() for scene in output["reg"])
            ground_truth.extend(scene[0].numpy() for scene in batch["gt_preds"])
            valid.extend(scene[0].numpy() for scene in batch["has_preds"])

    predictions = np.asarray(predictions, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    metrics = {}
    for k in (1, experiment.num_modes):
        values = displacement_metrics(
            predictions,
            ground_truth,
            valid,
            horizon=experiment.pred_len,
            k=k,
        )
        values.pop("best_mode")
        metrics[f"K={k}"] = values
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
