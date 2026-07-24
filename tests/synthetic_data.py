"""Synthetic trajectory frames used only for code-chain validation."""

import numpy as np
import pandas as pd

from timeba.data.registry import get_dataset_schema


def make_synthetic_frame(experiment, num_actors=2):
    """Create complete deterministic trajectories for a registered schema."""
    schema = get_dataset_schema(experiment.dataset)
    rows = []
    for actor_index in range(num_actors):
        actor_id = actor_index + 1
        actor_type = schema.agent_label if actor_index == 0 else "OTHERS"
        for step in range(experiment.total_len):
            row = {
                schema.frame_column: step,
                schema.actor_id_column: actor_id,
                schema.actor_type_column: actor_type,
            }
            for column_index, column in enumerate(schema.available_columns):
                if column in schema.identity_columns:
                    continue
                row[column] = np.float32(
                    0.01 * (column_index + 1)
                    + 0.1 * actor_index
                    + 0.02 * step
                )
            x_column, y_column = schema.feature_groups["position"]
            row[x_column] = np.float32(step * 0.5 + actor_index * 2.0)
            row[y_column] = np.float32(actor_index * 1.5 + step * 0.1)
            rows.append(row)
    return pd.DataFrame(rows)
