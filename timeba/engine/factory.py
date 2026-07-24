"""Build the canonical execution chain from one experiment configuration."""

from dataclasses import dataclass

import torch

from timeba import Timeba
from timeba.data.features import ResolvedFeatureSpec, resolve_feature_spec
from timeba.data.registry import get_dataset_schema
from timeba.data.schemas import DatasetSchema
from timeba.losses import Loss

from .optimizer import PiecewiseConstantLR, build_optimizer


@dataclass
class PipelineComponents:
    experiment: object
    schema: DatasetSchema
    features: ResolvedFeatureSpec
    model: Timeba
    loss: Loss
    optimizer: object
    lr_schedule: PiecewiseConstantLR
    device: torch.device


def build_pipeline(experiment, device="cuda"):
    """Build schema -> features -> Timeba -> Loss -> Adam."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "the frozen canonical Timeba runtime currently requires CUDA"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the canonical pipeline")

    schema = get_dataset_schema(experiment.dataset)
    features = resolve_feature_spec(schema, experiment.feature_spec)
    model = Timeba(
        input_dim=features.input_dim,
        pred_len=experiment.pred_len,
        num_modes=experiment.num_modes,
    ).to(device)
    loss = Loss(experiment.loss_config()).to(device)
    optimizer = build_optimizer(model.parameters(), experiment.training)
    lr_schedule = PiecewiseConstantLR(
        learning_rates=experiment.training.learning_rates,
        milestones=experiment.training.lr_milestones,
    )
    return PipelineComponents(
        experiment=experiment,
        schema=schema,
        features=features,
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        device=device,
    )
