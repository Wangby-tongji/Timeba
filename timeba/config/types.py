"""Typed experiment configuration independent of model implementation."""

from dataclasses import dataclass, field
from typing import Tuple

from timeba.data.features import FeatureSpec


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings preserved from the historical experiment interface."""

    epochs: int = 52
    batch_size: int = 16
    optimizer: str = "adam"
    learning_rates: Tuple[float, ...] = (1e-3, 1e-4, 4e-5)
    lr_milestones: Tuple[int, ...] = (32, 42)
    num_workers: int = 0

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.optimizer != "adam":
            raise ValueError("the canonical pipeline currently supports Adam")
        if len(self.learning_rates) != len(self.lr_milestones) + 1:
            raise ValueError(
                "learning_rates must contain one more value than lr_milestones"
            )
        if tuple(sorted(self.lr_milestones)) != self.lr_milestones:
            raise ValueError("lr_milestones must be sorted")


@dataclass(frozen=True)
class ExperimentConfig:
    """Dataset, feature, sequence, loss, and training configuration."""

    name: str
    dataset: str
    history_len: int
    pred_len: int
    feature_spec: FeatureSpec
    num_modes: int = 6
    pred_range: Tuple[float, float, float, float] = (
        -100.0,
        100.0,
        -100.0,
        100.0,
    )
    rotation_augmentation: bool = False
    cls_coef: float = 1.0
    reg_coef: float = 1.0
    margin: float = 0.2
    cls_threshold: float = 2.0
    cls_ignore: float = 0.2
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if not self.name:
            raise ValueError("experiment name must not be empty")
        if self.history_len <= 1:
            raise ValueError("history_len must be greater than one")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if self.history_len % 8 != 0:
            raise ValueError(
                "history_len must be divisible by 8 for Timeba alignment"
            )

    @property
    def total_len(self) -> int:
        return self.history_len + self.pred_len

    def loss_config(self):
        """Return the exact dictionary contract consumed by historical Loss."""
        return {
            "num_mods": self.num_modes,
            "num_preds": self.pred_len,
            "cls_coef": self.cls_coef,
            "reg_coef": self.reg_coef,
            "mgn": self.margin,
            "cls_th": self.cls_threshold,
            "cls_ignore": self.cls_ignore,
        }
