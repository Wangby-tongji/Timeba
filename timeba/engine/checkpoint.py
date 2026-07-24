"""Strict, atomic checkpoints with legacy-compatible core keys."""

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(path, components, epoch, extra=None):
    """Atomically save without overwriting an existing result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "state_dict": _cpu_state_dict(components.model),
        "opt_state": components.optimizer.state_dict(),
        "experiment": asdict(components.experiment),
    }
    if extra is not None:
        payload["extra"] = extra

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_checkpoint(
    path,
    model,
    optimizer=None,
    strict=True,
    map_location="cpu",
):
    """Load model strictly by default and optionally restore Adam state."""
    payload = torch.load(path, map_location=map_location)
    if "state_dict" not in payload:
        raise KeyError("checkpoint does not contain state_dict")
    incompatible = model.load_state_dict(payload["state_dict"], strict=strict)
    if optimizer is not None:
        if "opt_state" not in payload:
            raise KeyError("checkpoint does not contain opt_state")
        optimizer.load_state_dict(payload["opt_state"])
    return payload, incompatible
