"""Evaluation output extraction driven by ExperimentConfig.pred_len."""

import numpy as np
import torch

from timeba.evaluation.metrics import displacement_metrics


def evaluate_batch(components, batch):
    """Evaluate the first (AGENT) actor in each scene."""
    components.model.eval()
    with torch.no_grad():
        output = components.model(dict(batch))

    cls = torch.cat([scene[0:1] for scene in output["cls"]], dim=0)
    reg = torch.cat([scene[0:1] for scene in output["reg"]], dim=0)
    gt_preds = torch.cat(
        [scene[0:1] for scene in batch["gt_preds"]],
        dim=0,
    )
    has_preds = torch.cat(
        [scene[0:1] for scene in batch["has_preds"]],
        dim=0,
    )
    expected_reg = (
        len(output["reg"]),
        components.experiment.num_modes,
        components.experiment.pred_len,
        2,
    )
    if tuple(reg.shape) != expected_reg:
        raise ValueError(
            f"unexpected trajectory output shape {tuple(reg.shape)}; "
            f"expected {expected_reg}"
        )

    predictions = reg.detach().cpu().numpy()
    ground_truth = gt_preds.detach().cpu().numpy()
    valid = has_preds.detach().cpu().numpy()
    metrics = {}
    for k in (1, components.experiment.num_modes):
        metrics[f"K={k}"] = displacement_metrics(
            predictions,
            ground_truth,
            valid,
            horizon=components.experiment.pred_len,
            k=k,
        )
    return {
        "cls": cls,
        "reg": reg,
        "gt_preds": gt_preds,
        "has_preds": has_preds,
        "metrics": metrics,
    }
