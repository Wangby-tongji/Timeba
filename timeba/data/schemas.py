"""Explicit declarations of attributes available in each historical dataset."""

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class DatasetSchema:
    """Available raw attributes and identity columns for one dataset."""

    name: str
    feature_groups: Mapping[str, Tuple[str, ...]]
    frame_column: str = "frame"
    actor_id_column: str = "id"
    actor_type_column: str = "type"
    agent_label: str = "AGENT"

    @property
    def identity_columns(self):
        return (
            self.frame_column,
            self.actor_id_column,
            self.actor_type_column,
        )

    @property
    def available_columns(self):
        columns = list(self.identity_columns)
        for group_columns in self.feature_groups.values():
            for column in group_columns:
                if column not in columns:
                    columns.append(column)
        return tuple(columns)


NGSIM_SCHEMA = DatasetSchema(
    name="ngsim",
    feature_groups={
        "position": ("x", "y"),
        "dimensions": ("v_Width", "v_Length"),
        "local_position": ("Local_X", "Local_Y"),
        "velocity": ("v_Vel",),
        "acceleration": ("v_Acc",),
        "lane": ("Lane_ID",),
        "space_headway": ("Space_Headway",),
        "time_headway": ("Time_Headway",),
        "vehicle_class": ("v_Class",),
    },
)

HIGHD_SCHEMA = DatasetSchema(
    name="highd",
    feature_groups={
        "position": ("x", "y"),
        "dimensions": ("width", "height"),
        "velocity": ("xVelocity", "yVelocity"),
        "acceleration": ("xAcceleration", "yAcceleration"),
        "sight_distance": ("frontSightDistance", "backSightDistance"),
        "distance_headway": ("dhw",),
        "time_headway": ("thw",),
        "time_to_collision": ("ttc",),
        "preceding_velocity": ("precedingXVelocity",),
    },
)

EXID_SCHEMA = DatasetSchema(
    name="exid",
    feature_groups={
        "position": ("x", "y"),
        "heading": ("heading",),
        "dimensions": ("width", "length"),
        "velocity": ("xVelocity", "yVelocity"),
        "acceleration": ("xAcceleration", "yAcceleration"),
        "lane_velocity": ("lonVelocity", "latVelocity"),
        "lane_acceleration": ("lonAcceleration", "latAcceleration"),
        "lead_distance_headway": ("leadDHW",),
        "lead_time_headway": ("leadTHW",),
        "lead_time_to_collision": ("leadTTC",),
        "vehicle_class": ("class2",),
    },
)

IND_SCHEMA = DatasetSchema(
    name="ind",
    feature_groups={
        "position": ("x", "y"),
        "heading": ("heading",),
        "dimensions": ("width", "length"),
        "velocity": ("xVelocity", "yVelocity"),
        "acceleration": ("xAcceleration", "yAcceleration"),
        "lane_velocity": ("lonVelocity", "latVelocity"),
        "lane_acceleration": ("lonAcceleration", "latAcceleration"),
        "vehicle_class": ("class2",),
    },
)


DATASET_SCHEMAS = {
    schema.name: schema
    for schema in (NGSIM_SCHEMA, HIGHD_SCHEMA, EXID_SCHEMA, IND_SCHEMA)
}
