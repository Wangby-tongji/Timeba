"""Dependency-free source checks for the three historical full models."""

import ast
import unittest
from pathlib import Path

from tests.compatibility.golden_manifest import (
    ORDERED_STATE_DICT,
    ORDERED_STATE_DICT_KEYS,
    SOURCE_COMMIT,
    TOTAL_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FULL_MODEL_FILES = (
    "NGSIM24_5_4.py",
    "highD3_5_9.py",
    "exiD3_5_6.py",
)
SHARED_CLASSES = (
    "Net",
    "A2A",
    "PredNet",
    "Linear_dev",
    "Att",
    "GLU",
    "GatedResdualNetwork",
    "AttDest",
    "PredLoss",
    "Loss",
)


def _tree(filename):
    return ast.parse((REPOSITORY_ROOT / filename).read_text())


def _class_node(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


class _NormalizeHistoricalInputDim(ast.NodeTransformer):
    """Replace only the initial hard-coded ActorNet n_in value."""

    def visit_Assign(self, node):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "n_in"
            and isinstance(node.value, ast.Constant)
        ):
            node.value = ast.Constant(value="<historical-input-dim>")
        return self.generic_visit(node)


class HistoricalFullModelSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = {name: _tree(name) for name in FULL_MODEL_FILES}

    def test_shared_architecture_classes_are_identical(self):
        reference = self.trees[FULL_MODEL_FILES[0]]
        for class_name in SHARED_CLASSES:
            expected = ast.dump(
                _class_node(reference, class_name),
                include_attributes=False,
            )
            for filename in FULL_MODEL_FILES[1:]:
                actual = ast.dump(
                    _class_node(self.trees[filename], class_name),
                    include_attributes=False,
                )
                self.assertEqual(
                    expected,
                    actual,
                    f"{class_name} differs in {filename}",
                )

    def test_actor_net_differs_only_by_historical_input_dim(self):
        normalized = []
        for filename in FULL_MODEL_FILES:
            actor_net = _class_node(self.trees[filename], "ActorNet")
            actor_net = _NormalizeHistoricalInputDim().visit(
                ast.fix_missing_locations(actor_net)
            )
            normalized.append(ast.dump(actor_net, include_attributes=False))
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[0], normalized[2])

    def test_copied_component_classes_preserve_historical_ast(self):
        comparisons = (
            (
                "layers.py",
                "timeba/models/blocks.py",
                (
                    "Unet1d",
                    "Conv1d",
                    "Linear",
                    "MambaBlock",
                    "GroupNorm",
                    "LinearRes2",
                    "LinearRes",
                ),
            ),
            (
                "NGSIM24_5_4.py",
                "timeba/models/interaction.py",
                (
                    "A2A",
                    "EncodeDist",
                    "Linear_dev",
                    "Att",
                    "GLU",
                    "GatedResdualNetwork",
                ),
            ),
            (
                "NGSIM24_5_4.py",
                "timeba/models/prediction.py",
                ("PredNet", "AttDest"),
            ),
            (
                "NGSIM24_5_4.py",
                "timeba/losses.py",
                ("PredLoss", "Loss"),
            ),
        )
        for old_file, new_file, class_names in comparisons:
            old_tree = _tree(old_file)
            new_tree = _tree(new_file)
            for class_name in class_names:
                self.assertEqual(
                    ast.dump(
                        _class_node(old_tree, class_name),
                        include_attributes=False,
                    ),
                    ast.dump(
                        _class_node(new_tree, class_name),
                        include_attributes=False,
                    ),
                    f"{class_name} changed while copied to {new_file}",
                )


class GoldenManifestIntegrityTest(unittest.TestCase):
    def test_manifest_identity_and_counts(self):
        self.assertEqual(
            SOURCE_COMMIT,
            "1114c0dd976b07914010827ddd26f4f43f9d70e6",
        )
        self.assertEqual(len(ORDERED_STATE_DICT), 316)
        self.assertEqual(len(set(ORDERED_STATE_DICT_KEYS)), 316)
        self.assertEqual(TOTAL_PARAMETER_COUNT, 38_629_465)
        self.assertEqual(TRAINABLE_PARAMETER_COUNT, 38_629_465)

    def test_checkpoint_sensitive_unused_parameters_are_present(self):
        required_keys = {
            "actor_net.groups.0.0.conv2.weight",
            "actor_net.lateral.0.conv.weight",
            "actor_net.Unet.3.up.weight",
            "actor_net.output.conv2.weight",
        }
        self.assertTrue(required_keys.issubset(ORDERED_STATE_DICT_KEYS))


if __name__ == "__main__":
    unittest.main()
