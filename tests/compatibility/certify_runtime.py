"""Run Stage 2B checkpoint and numerical equivalence certification."""

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

from tests.compatibility.golden_manifest import (
    HISTORY_LEN,
    INPUT_DIM,
    NUM_MODES,
    ORDERED_STATE_DICT_KEYS,
    PRED_LEN,
    STATE_DICT_SHAPES,
    TOTAL_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
)
from tests.compatibility.runtime_manifest import (
    DEFAULT_OUTPUT as DEFAULT_MANIFEST_OUTPUT,
    environment_manifest,
    legacy_config,
    state_dict_manifest,
    write_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_OUTPUT = (
    REPOSITORY_ROOT
    / "tests"
    / "compatibility"
    / "runtime_certification.json"
)
RTOL = 1e-5
ATOL = 1e-6
SEED = 0


def _clone_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(torch, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_tree(torch, item) for item in value]
    return value


def _tensor_pairs(torch, legacy, canonical, path="output"):
    if isinstance(legacy, torch.Tensor):
        if not isinstance(canonical, torch.Tensor):
            raise AssertionError(f"{path}: canonical value is not a tensor")
        yield path, legacy, canonical
        return

    if isinstance(legacy, dict):
        if not isinstance(canonical, dict):
            raise AssertionError(f"{path}: canonical value is not a dict")
        if list(legacy.keys()) != list(canonical.keys()):
            raise AssertionError(
                f"{path}: dict keys differ: "
                f"{list(legacy.keys())} != {list(canonical.keys())}"
            )
        for key in legacy:
            yield from _tensor_pairs(
                torch,
                legacy[key],
                canonical[key],
                f"{path}.{key}",
            )
        return

    if isinstance(legacy, (list, tuple)):
        if not isinstance(canonical, (list, tuple)):
            raise AssertionError(f"{path}: canonical value is not a sequence")
        if len(legacy) != len(canonical):
            raise AssertionError(
                f"{path}: sequence lengths differ: "
                f"{len(legacy)} != {len(canonical)}"
            )
        for index, (legacy_item, canonical_item) in enumerate(
            zip(legacy, canonical)
        ):
            yield from _tensor_pairs(
                torch,
                legacy_item,
                canonical_item,
                f"{path}[{index}]",
            )
        return

    if legacy != canonical:
        raise AssertionError(f"{path}: values differ: {legacy} != {canonical}")


def _compare_tree(torch, legacy, canonical):
    maximum = 0.0
    compared = 0
    for path, legacy_tensor, canonical_tensor in _tensor_pairs(
        torch, legacy, canonical
    ):
        try:
            torch.testing.assert_close(
                legacy_tensor,
                canonical_tensor,
                rtol=RTOL,
                atol=ATOL,
            )
        except AssertionError as exc:
            raise AssertionError(f"{path}: {exc}") from exc
        if legacy_tensor.numel():
            difference = (
                (legacy_tensor - canonical_tensor)
                .abs()
                .max()
                .detach()
                .cpu()
                .item()
            )
            maximum = max(maximum, float(difference))
        compared += 1
    return {"tensor_count": compared, "maximum_absolute_difference": maximum}


def _comparison_result(torch, legacy, canonical):
    try:
        return {
            "passed": True,
            **_compare_tree(torch, legacy, canonical),
        }
    except AssertionError as exc:
        maximum = None
        compared = 0
        try:
            differences = []
            for _path, legacy_tensor, canonical_tensor in _tensor_pairs(
                torch, legacy, canonical
            ):
                compared += 1
                if legacy_tensor.numel():
                    differences.append(
                        float(
                            (legacy_tensor - canonical_tensor)
                            .abs()
                            .max()
                            .detach()
                            .cpu()
                            .item()
                        )
                    )
            maximum = max(differences, default=0.0)
        except Exception:
            pass
        return {
            "passed": False,
            "tensor_count": compared,
            "maximum_absolute_difference": maximum,
            "error": str(exc),
        }


def _source_expectation_comparison(runtime_model_manifest):
    entries = runtime_model_manifest["ordered_state_dict"]
    runtime_keys = tuple(entry["key"] for entry in entries)
    runtime_shapes = {
        entry["key"]: tuple(entry["shape"]) for entry in entries
    }
    return {
        "ordered_keys_match": runtime_keys == ORDERED_STATE_DICT_KEYS,
        "shapes_match": runtime_shapes == STATE_DICT_SHAPES,
        "total_parameter_count_match": (
            runtime_model_manifest["total_parameter_count"]
            == TOTAL_PARAMETER_COUNT
        ),
        "trainable_parameter_count_match": (
            runtime_model_manifest["trainable_parameter_count"]
            == TRAINABLE_PARAMETER_COUNT
        ),
        "source_derived_expected_entry_count": len(
            ORDERED_STATE_DICT_KEYS
        ),
        "source_derived_expected_total_parameters": TOTAL_PARAMETER_COUNT,
        "source_derived_expected_trainable_parameters": (
            TRAINABLE_PARAMETER_COUNT
        ),
    }


def _dummy_data(torch):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return {
        "feats": [
            torch.randn(
                2,
                HISTORY_LEN,
                INPUT_DIM,
                generator=generator,
            ),
            torch.randn(
                1,
                HISTORY_LEN,
                INPUT_DIM,
                generator=generator,
            ),
        ],
        "ctrs": [
            torch.tensor([[0.0, 0.0], [3.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, -2.0]], dtype=torch.float32),
        ],
        "rot": [
            torch.eye(2, dtype=torch.float32),
            torch.eye(2, dtype=torch.float32),
        ],
        "orig": [
            torch.tensor([10.0, 20.0], dtype=torch.float32),
            torch.tensor([-4.0, 2.0], dtype=torch.float32),
        ],
    }


def _install_stage_hooks(torch, model, capture):
    handles = []

    def register(name, module):
        def hook(_module, _inputs, output):
            capture[name] = _clone_tree(torch, output)

        handles.append(module.register_forward_hook(hook))

    register("actor_net", model.actor_net)
    register("a2a", model.a2a)
    register("pred_net", model.pred_net)
    return handles


def _remove_hooks(handles):
    for handle in handles:
        handle.remove()


def _write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output


def certify(manifest_output, report_output):
    torch = importlib.import_module("torch")
    mamba_ssm = importlib.import_module("mamba_ssm")
    legacy_module = importlib.import_module("NGSIM24_5_4")
    canonical_module = importlib.import_module("timeba.models.timeba")

    report = {
        "certification": "pending",
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "seed": SEED,
        "environment": environment_manifest(torch, mamba_ssm),
    }

    legacy = legacy_module.Net(legacy_config())
    canonical = canonical_module.Timeba(
        input_dim=INPUT_DIM,
        pred_len=PRED_LEN,
        num_modes=NUM_MODES,
    )

    runtime_model_manifest = state_dict_manifest(legacy)
    runtime_manifest = {
        "manifest_type": "runtime-derived-legacy-state-dict",
        "source_module": "NGSIM24_5_4.Net",
        "model_arguments": {
            "input_dim": INPUT_DIM,
            "history_len": HISTORY_LEN,
            "pred_len": PRED_LEN,
            "num_modes": NUM_MODES,
        },
        "environment": report["environment"],
        "model": runtime_model_manifest,
    }
    write_manifest(runtime_manifest, manifest_output)

    legacy_state = legacy.state_dict()
    canonical_state = canonical.state_dict()
    legacy_keys = list(legacy_state.keys())
    canonical_keys = list(canonical_state.keys())
    report["runtime_legacy_manifest"] = {
        "output": str(Path(manifest_output)),
        "state_dict_entry_count": len(legacy_keys),
        "total_parameter_count": runtime_model_manifest[
            "total_parameter_count"
        ],
        "trainable_parameter_count": runtime_model_manifest[
            "trainable_parameter_count"
        ],
    }
    report["source_expectation_diagnostic"] = (
        _source_expectation_comparison(runtime_model_manifest)
    )

    report["ordered_key_parity"] = legacy_keys == canonical_keys
    report["key_set_parity"] = set(legacy_keys) == set(canonical_keys)
    shape_mismatches = {
        key: {
            "legacy": list(legacy_state[key].shape),
            "canonical": list(canonical_state[key].shape),
        }
        for key in set(legacy_keys).intersection(canonical_keys)
        if legacy_state[key].shape != canonical_state[key].shape
    }
    report["shape_parity"] = (
        report["key_set_parity"] and not shape_mismatches
    )
    report["shape_mismatches"] = shape_mismatches

    legacy_total = sum(
        parameter.numel() for parameter in legacy.parameters()
    )
    canonical_total = sum(
        parameter.numel() for parameter in canonical.parameters()
    )
    legacy_trainable = sum(
        parameter.numel()
        for parameter in legacy.parameters()
        if parameter.requires_grad
    )
    canonical_trainable = sum(
        parameter.numel()
        for parameter in canonical.parameters()
        if parameter.requires_grad
    )
    report["parameter_counts"] = {
        "legacy_total": legacy_total,
        "canonical_total": canonical_total,
        "legacy_trainable": legacy_trainable,
        "canonical_trainable": canonical_trainable,
    }
    report["total_parameter_count_parity"] = (
        legacy_total == canonical_total
    )
    report["trainable_parameter_count_parity"] = (
        legacy_trainable == canonical_trainable
    )

    strict_results = {}
    try:
        result = canonical.load_state_dict(legacy_state, strict=True)
        strict_results["legacy_to_canonical"] = {
            "passed": True,
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
        }
    except Exception as exc:
        strict_results["legacy_to_canonical"] = {
            "passed": False,
            "error": repr(exc),
        }

    try:
        result = legacy.load_state_dict(canonical.state_dict(), strict=True)
        strict_results["canonical_to_legacy"] = {
            "passed": True,
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
        }
    except Exception as exc:
        strict_results["canonical_to_legacy"] = {
            "passed": False,
            "error": repr(exc),
        }
    report["strict_loading"] = strict_results

    if not torch.cuda.is_available():
        report["forward_parity"] = {
            "status": "blocked",
            "reason": "Historical forward requires CUDA via gpu()/.cuda()",
        }
    elif not all(item["passed"] for item in strict_results.values()):
        report["forward_parity"] = {
            "status": "not-run",
            "reason": "strict state_dict loading failed",
        }
    else:
        legacy = legacy.cuda().eval()
        canonical = canonical.cuda().eval()
        canonical.load_state_dict(legacy.state_dict(), strict=True)

        data = _dummy_data(torch)
        legacy_actors, legacy_idcs = legacy_module.actor_gather(data["feats"])
        canonical_actors, canonical_idcs = canonical_module.actor_gather(
            data["feats"]
        )
        actor_idcs_equal = len(legacy_idcs) == len(canonical_idcs) and all(
            torch.equal(old, new)
            for old, new in zip(legacy_idcs, canonical_idcs)
        )
        report["dummy_input"] = {
            "device": "cuda:0",
            "dtype": "torch.float32",
            "input_dim": INPUT_DIM,
            "history_len": HISTORY_LEN,
            "pred_len": PRED_LEN,
            "num_modes": NUM_MODES,
            "actor_idcs_equal": actor_idcs_equal,
            "gathered_actor_shape_equal": (
                legacy_actors.shape == canonical_actors.shape
            ),
            "actor_ctrs_shared_source": True,
        }

        captures = {"legacy": {}, "canonical": {}}
        legacy_handles = _install_stage_hooks(
            torch, legacy, captures["legacy"]
        )
        canonical_handles = _install_stage_hooks(
            torch, canonical, captures["canonical"]
        )
        try:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            with torch.no_grad():
                legacy_out = legacy(_clone_tree(torch, data))

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            with torch.no_grad():
                canonical_out = canonical(_clone_tree(torch, data))
        finally:
            _remove_hooks(legacy_handles)
            _remove_hooks(canonical_handles)

        diagnostics = {}
        first_divergent_stage = None
        for stage in ("actor_net", "a2a", "pred_net"):
            diagnostics[stage] = _comparison_result(
                torch,
                captures["legacy"][stage],
                captures["canonical"][stage],
            )
            if (
                not diagnostics[stage]["passed"]
                and first_divergent_stage is None
            ):
                first_divergent_stage = stage

        final_comparison = _comparison_result(
            torch, legacy_out, canonical_out
        )
        final_result = {
            "status": (
                "passed" if final_comparison["passed"] else "failed"
            ),
            **final_comparison,
        }
        if not final_comparison["passed"]:
            if first_divergent_stage is None:
                first_divergent_stage = "final_output"

        classification_comparison = _comparison_result(
            torch,
            legacy_out["cls"],
            canonical_out["cls"],
        )
        regression_comparison = _comparison_result(
            torch,
            legacy_out["reg"],
            canonical_out["reg"],
        )
        final_result["classification"] = classification_comparison
        final_result["regression"] = regression_comparison
        final_result["intermediate_diagnostics"] = diagnostics
        final_result["first_divergent_stage"] = first_divergent_stage
        report["forward_parity"] = final_result

    required_results = (
        report["ordered_key_parity"],
        report["shape_parity"],
        report["total_parameter_count_parity"],
        report["trainable_parameter_count_parity"],
        all(item["passed"] for item in strict_results.values()),
        report["forward_parity"].get("status") == "passed",
        report.get("dummy_input", {}).get("actor_idcs_equal", False),
        report.get("dummy_input", {}).get(
            "gathered_actor_shape_equal", False
        ),
    )
    report["certification"] = (
        "equivalent" if all(required_results) else "not-certified"
    )
    _write_report(report, report_output)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Certify canonical Timeba against NGSIM24_5_4"
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
    )
    args = parser.parse_args()

    try:
        report = certify(args.manifest_output, args.report_output)
    except Exception:
        traceback.print_exc()
        return 2

    print(json.dumps(report, indent=2))
    return 0 if report["certification"] == "equivalent" else 1


if __name__ == "__main__":
    sys.exit(main())
