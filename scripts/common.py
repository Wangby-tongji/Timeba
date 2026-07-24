"""Shared CLI helpers for canonical experiments and prepared CSV splits."""

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from timeba.config import get_experiment_config
from timeba.data import TrajectoryDataset, collate_trajectory_batch
from timeba.data.registry import get_dataset_schema


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--experiment",
        required=True,
        choices=("ngsim_paper", "highd_paper", "exid_paper"),
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="directory containing prepared train/val/test CSV split folders",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, default=0)


def resolve_experiment(args):
    experiment = get_experiment_config(args.experiment)
    training = experiment.training
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    if args.num_workers is not None:
        training = replace(training, num_workers=args.num_workers)
    return replace(experiment, training=training)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_split_loader(experiment, data_root, split, train=False):
    schema = get_dataset_schema(experiment.dataset)
    dataset = TrajectoryDataset.from_directory(
        Path(data_root) / split,
        schema,
        experiment,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=experiment.training.batch_size,
        num_workers=experiment.training.num_workers,
        collate_fn=collate_trajectory_batch,
        pin_memory=True,
        shuffle=False,
        drop_last=train,
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def experiment_dict(experiment):
    return asdict(experiment)
