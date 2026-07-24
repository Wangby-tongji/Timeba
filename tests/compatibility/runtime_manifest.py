"""Generate a weight-free manifest directly from the legacy runtime model."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tests.compatibility.golden_manifest import (
    HISTORY_LEN,
    INPUT_DIM,
    NUM_MODES,
    PRED_LEN,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "tests"
    / "compatibility"
    / "runtime_legacy_manifest.json"
)


def legacy_config():
    return {
        "n_actor": 512,
        "actor2actor_dist": 100.0,
        "num_preds": PRED_LEN,
        "num_mods": NUM_MODES,
    }


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision(reference):
    try:
        result = subprocess.run(
            ["git", "rev-parse", reference],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_manifest(torch, mamba_ssm):
    cuda_available = torch.cuda.is_available()
    gpu_names = []
    if cuda_available:
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]

    cudnn_version = None
    if getattr(torch.backends, "cudnn", None) is not None:
        cudnn_version = torch.backends.cudnn.version()

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_available": cuda_available,
        "gpu_names": gpu_names,
        "mamba_ssm_version": (
            _distribution_version("mamba-ssm")
            or getattr(mamba_ssm, "__version__", None)
        ),
        "mamba_ssm_module": getattr(mamba_ssm, "__file__", None),
        "repository_head": _git_revision("HEAD"),
        "original_backup_commit": _git_revision("orin"),
    }


def state_dict_manifest(model):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    entries = []

    for index, (key, tensor) in enumerate(model.state_dict().items()):
        if key in parameters:
            kind = "parameter"
            requires_grad = parameters[key].requires_grad
        elif key in buffers:
            kind = "buffer"
            requires_grad = False
        else:
            # Retain unexpected state entries rather than silently dropping
            # them.  The runtime manifest is the source of truth.
            kind = "other"
            requires_grad = False

        entries.append(
            {
                "index": index,
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "kind": kind,
                "requires_grad": requires_grad,
            }
        )

    return {
        "state_dict_entry_count": len(entries),
        "ordered_state_dict": entries,
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def build_runtime_legacy_manifest():
    torch = importlib.import_module("torch")
    mamba_ssm = importlib.import_module("mamba_ssm")
    legacy_module = importlib.import_module("NGSIM24_5_4")
    legacy_model = legacy_module.Net(legacy_config())

    return {
        "manifest_type": "runtime-derived-legacy-state-dict",
        "source_module": "NGSIM24_5_4.Net",
        "model_arguments": {
            "input_dim": INPUT_DIM,
            "history_len": HISTORY_LEN,
            "pred_len": PRED_LEN,
            "num_modes": NUM_MODES,
        },
        "environment": environment_manifest(torch, mamba_ssm),
        "model": state_dict_manifest(legacy_model),
    }


def write_manifest(manifest, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate the runtime legacy Timeba state_dict manifest"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = build_runtime_legacy_manifest()
    output = write_manifest(manifest, args.output)
    print(output)


if __name__ == "__main__":
    main()
