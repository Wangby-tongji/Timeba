"""Canonical checkpoint layout regression against the certified golden manifest."""

import importlib
import unittest

from tests.compatibility.golden_manifest import (
    INPUT_DIM,
    NUM_MODES,
    ORDERED_STATE_DICT_KEYS,
    PRED_LEN,
    SOURCE_COMMIT,
    STATE_DICT_SHAPES,
    TOTAL_PARAMETER_COUNT,
)


def _runtime():
    try:
        torch = importlib.import_module("torch")
        Timeba = importlib.import_module(
            "timeba.models.timeba"
        ).Timeba
    except (ImportError, OSError) as exc:
        raise unittest.SkipTest(
            f"PyTorch/Mamba runtime unavailable: {exc}"
        ) from exc
    return torch, Timeba


class CanonicalManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, Timeba = _runtime()
        cls.model = Timeba(
            input_dim=INPUT_DIM,
            pred_len=PRED_LEN,
            num_modes=NUM_MODES,
        )

    def test_certified_manifest_identity(self):
        self.assertEqual(
            SOURCE_COMMIT,
            "1114c0dd976b07914010827ddd26f4f43f9d70e6",
        )
        self.assertEqual(len(ORDERED_STATE_DICT_KEYS), 316)
        self.assertEqual(TOTAL_PARAMETER_COUNT, 38_629_465)

    def test_ordered_keys_and_shapes(self):
        state = self.model.state_dict()
        self.assertEqual(tuple(state), ORDERED_STATE_DICT_KEYS)
        for key, tensor in state.items():
            self.assertEqual(tuple(tensor.shape), STATE_DICT_SHAPES[key], key)

    def test_parameter_count(self):
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(total, TOTAL_PARAMETER_COUNT)
        self.assertEqual(trainable, TOTAL_PARAMETER_COUNT)

    def test_strict_round_trip_between_canonical_instances(self):
        Timeba = type(self.model)
        reloaded = Timeba(
            input_dim=INPUT_DIM,
            pred_len=PRED_LEN,
            num_modes=NUM_MODES,
        )
        incompatible = reloaded.load_state_dict(
            self.model.state_dict(),
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_checkpoint_sensitive_registered_parameters_remain(self):
        required = {
            "actor_net.groups.0.0.conv2.weight",
            "actor_net.lateral.0.conv.weight",
            "actor_net.Unet.3.up.weight",
            "actor_net.output.conv2.weight",
        }
        self.assertTrue(required.issubset(self.model.state_dict()))


if __name__ == "__main__":
    unittest.main()
