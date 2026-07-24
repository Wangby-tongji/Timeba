"""Configuration contracts and canonical experiment presets."""

from .presets import (
    EXID_PAPER,
    HIGHD_PAPER,
    IND_HISTORICAL_EXTENSION,
    NGSIM_PAPER,
    get_experiment_config,
)
from .types import ExperimentConfig, TrainingConfig

__all__ = [
    "EXID_PAPER",
    "HIGHD_PAPER",
    "IND_HISTORICAL_EXTENSION",
    "NGSIM_PAPER",
    "ExperimentConfig",
    "TrainingConfig",
    "get_experiment_config",
]
