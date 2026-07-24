"""Runtime compatibility tests for legacy and canonical full Timeba."""

import importlib
import unittest

from tests.compatibility.golden_manifest import (
    NUM_MODES,
    ORDERED_STATE_DICT_KEYS,
    STATE_DICT_SHAPES,
    TOTAL_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
)


def _legacy_config():
    return {
        "n_actor": 512,
        "actor2actor_dist": 100.0,
        "num_preds": 50,
        "num_mods": 6,
    }


def _runtime_modules():
    try:
        torch = importlib.import_module("torch")
        legacy_module = importlib.import_module("NGSIM24_5_4")
        canonical_module = importlib.import_module("timeba.models.timeba")
    except (ImportError, OSError) as exc:
        raise unittest.SkipTest(
            f"PyTorch/Mamba compatibility runtime unavailable: {exc}"
        ) from exc
    return torch, legacy_module, canonical_module


class TimebaCheckpointCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, legacy_module, canonical_module = _runtime_modules()
        cls.legacy = legacy_module.Net(_legacy_config())
        cls.canonical = canonical_module.Timeba(
            input_dim=5,
            pred_len=50,
            num_modes=NUM_MODES,
        )

    def test_legacy_matches_golden_manifest(self):
        state = self.legacy.state_dict()
        self.assertEqual(tuple(state.keys()), ORDERED_STATE_DICT_KEYS)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in state.items()},
            STATE_DICT_SHAPES,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in self.legacy.parameters()),
            TOTAL_PARAMETER_COUNT,
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in self.legacy.parameters()
                if parameter.requires_grad
            ),
            TRAINABLE_PARAMETER_COUNT,
        )

    def test_state_dict_keys_order_and_shapes_are_identical(self):
        legacy_state = self.legacy.state_dict()
        canonical_state = self.canonical.state_dict()
        self.assertEqual(tuple(legacy_state.keys()), tuple(canonical_state.keys()))
        self.assertEqual(set(legacy_state), set(canonical_state))
        for key in legacy_state:
            self.assertEqual(
                tuple(legacy_state[key].shape),
                tuple(canonical_state[key].shape),
                key,
            )

    def test_parameter_counts_are_identical(self):
        legacy_total = sum(
            parameter.numel() for parameter in self.legacy.parameters()
        )
        canonical_total = sum(
            parameter.numel() for parameter in self.canonical.parameters()
        )
        legacy_trainable = sum(
            parameter.numel()
            for parameter in self.legacy.parameters()
            if parameter.requires_grad
        )
        canonical_trainable = sum(
            parameter.numel()
            for parameter in self.canonical.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(legacy_total, canonical_total)
        self.assertEqual(legacy_trainable, canonical_trainable)

    def test_legacy_state_loads_into_canonical_with_strict_true(self):
        result = self.canonical.load_state_dict(
            self.legacy.state_dict(),
            strict=True,
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_canonical_state_loads_into_legacy_with_strict_true(self):
        result = self.legacy.load_state_dict(
            self.canonical.state_dict(),
            strict=True,
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])


class TimebaForwardParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, legacy_module, canonical_module = _runtime_modules()
        if not cls.torch.cuda.is_available():
            raise unittest.SkipTest(
                "Historical forward calls .cuda(); CUDA is unavailable"
            )

        cls.torch.manual_seed(0)
        cls.torch.cuda.manual_seed_all(0)
        cls.legacy = legacy_module.Net(_legacy_config()).cuda().eval()
        cls.canonical = (
            canonical_module.Timeba(
                input_dim=5,
                pred_len=50,
                num_modes=NUM_MODES,
            )
            .cuda()
            .eval()
        )
        cls.canonical.load_state_dict(cls.legacy.state_dict(), strict=True)

    def _dummy_data(self):
        torch = self.torch
        return {
            "feats": [
                torch.randn(2, 8, 5),
                torch.randn(1, 8, 5),
            ],
            "ctrs": [
                torch.tensor([[0.0, 0.0], [3.0, 1.0]]),
                torch.tensor([[1.0, -2.0]]),
            ],
            "rot": [
                torch.eye(2),
                torch.eye(2),
            ],
            "orig": [
                torch.tensor([10.0, 20.0]),
                torch.tensor([-4.0, 2.0]),
            ],
        }

    def test_deterministic_forward_outputs_are_equivalent(self):
        data = self._dummy_data()
        with self.torch.no_grad():
            legacy_out = self.legacy(data)
            canonical_out = self.canonical(data)

        self.assertEqual(legacy_out.keys(), canonical_out.keys())
        for output_name in ("cls", "reg"):
            self.assertEqual(
                len(legacy_out[output_name]),
                len(canonical_out[output_name]),
            )
            for legacy_tensor, canonical_tensor in zip(
                legacy_out[output_name],
                canonical_out[output_name],
            ):
                self.torch.testing.assert_close(
                    legacy_tensor,
                    canonical_tensor,
                    rtol=1e-5,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
