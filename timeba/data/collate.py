"""Collation contract used by the historical list-per-scene model interface."""

import numpy as np
import torch


def _from_numpy(data):
    if isinstance(data, dict):
        return {key: _from_numpy(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_from_numpy(value) for value in data]
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data


def collate_trajectory_batch(batch):
    """Convert arrays to tensors and retain one list entry per scene."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    converted = _from_numpy(batch)
    keys = tuple(converted[0])
    for index, item in enumerate(converted[1:], start=1):
        if tuple(item) != keys:
            raise KeyError(
                f"batch item {index} has keys {tuple(item)}, expected {keys}"
            )
    return {key: [item[key] for item in converted] for key in keys}
