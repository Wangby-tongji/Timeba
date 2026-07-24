"""Semantic feature selection resolved to an exact ordered channel layout."""

from dataclasses import dataclass
from typing import Tuple

from .schemas import DatasetSchema


@dataclass(frozen=True)
class FeatureSpec:
    """Selected semantic groups; a group may resolve to multiple channels."""

    groups: Tuple[str, ...]
    include_presence: bool = True

    def __post_init__(self):
        if not self.groups:
            raise ValueError("at least one semantic feature group is required")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("feature groups must not be repeated")


@dataclass(frozen=True)
class ResolvedFeatureSpec:
    """Exact raw channel order and derived input dimension."""

    groups: Tuple[str, ...]
    raw_channels: Tuple[str, ...]
    position_indices: Tuple[int, int]
    include_presence: bool

    @property
    def input_dim(self):
        return len(self.raw_channels) + int(self.include_presence)

    @property
    def tensor_channels(self):
        channels = self.raw_channels
        if self.include_presence:
            channels += ("__presence__",)
        return channels


def resolve_feature_spec(schema: DatasetSchema, spec: FeatureSpec):
    """Resolve semantic groups without assuming one group is one channel."""
    unknown = [group for group in spec.groups if group not in schema.feature_groups]
    if unknown:
        raise KeyError(
            f"{schema.name} does not define feature groups: {', '.join(unknown)}"
        )
    if "position" not in spec.groups:
        raise ValueError("position must be selected for trajectory construction")

    raw_channels = []
    for group in spec.groups:
        raw_channels.extend(schema.feature_groups[group])
    if len(set(raw_channels)) != len(raw_channels):
        raise ValueError("resolved raw channels must be unique")

    position_columns = schema.feature_groups["position"]
    if len(position_columns) != 2:
        raise ValueError("position must resolve to exactly two raw channels")
    position_indices = tuple(raw_channels.index(column) for column in position_columns)
    if position_indices != (0, 1):
        raise ValueError(
            "position must be the first semantic group to preserve historical layout"
        )

    return ResolvedFeatureSpec(
        groups=spec.groups,
        raw_channels=tuple(raw_channels),
        position_indices=position_indices,
        include_presence=spec.include_presence,
    )
