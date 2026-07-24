"""K=1/K=all displacement metrics with a configured prediction horizon."""

import numpy as np


def displacement_metrics(predictions, ground_truth, has_preds, horizon, k):
    """Compute minFDE-selected ADE/FDE while preserving historical K semantics."""
    predictions = np.asarray(predictions, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    has_preds = np.asarray(has_preds, dtype=bool)

    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape (B, K, T, 2)")
    if ground_truth.shape != (
        predictions.shape[0],
        predictions.shape[2],
        2,
    ):
        raise ValueError("ground_truth must have shape (B, T, 2)")
    if has_preds.shape != predictions.shape[:1] + predictions.shape[2:3]:
        raise ValueError("has_preds must have shape (B, T)")
    if horizon <= 0 or horizon > predictions.shape[2]:
        raise ValueError("horizon exceeds available prediction steps")
    if k <= 0 or k > predictions.shape[1]:
        raise ValueError("k exceeds available prediction modes")

    predictions = predictions[:, :k, :horizon]
    ground_truth = ground_truth[:, :horizon]
    has_preds = has_preds[:, :horizon]
    valid_counts = has_preds.sum(axis=1)
    if np.any(valid_counts == 0):
        raise ValueError("every target must have at least one valid future step")

    errors = np.linalg.norm(
        predictions - ground_truth[:, None, :, :],
        axis=-1,
    )
    masked_errors = np.where(has_preds[:, None, :], errors, 0.0)
    last_indices = np.asarray(
        [np.flatnonzero(mask)[-1] for mask in has_preds],
        dtype=np.int64,
    )
    batch_indices = np.arange(len(predictions))
    final_errors = errors[
        batch_indices[:, None],
        np.arange(k)[None, :],
        last_indices[:, None],
    ]
    best_modes = final_errors.argmin(axis=1)
    best_errors = masked_errors[batch_indices, best_modes]

    ade = (best_errors.sum(axis=1) / valid_counts).mean()
    fde = errors[batch_indices, best_modes, last_indices].mean()
    miss_rate = (
        errors[batch_indices, best_modes, last_indices] > 2.0
    ).mean()
    return {
        "minADE": float(ade),
        "minFDE": float(fde),
        "MR": float(miss_rate),
        "best_mode": best_modes,
    }
