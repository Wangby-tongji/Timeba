import unittest

from timeba.config.presets import (
    EXID_PAPER,
    HIGHD_PAPER,
    NGSIM_PAPER,
)
from timeba.data.features import FeatureSpec, resolve_feature_spec
from timeba.data.registry import get_dataset_schema


class SchemaAndFeatureSpecTest(unittest.TestCase):
    def test_canonical_channel_orders(self):
        expected = {
            "ngsim_paper": (
                ("x", "y", "v_Vel", "v_Acc", "__presence__"),
                5,
            ),
            "highd_paper": (
                (
                    "x",
                    "y",
                    "width",
                    "height",
                    "xVelocity",
                    "yVelocity",
                    "xAcceleration",
                    "yAcceleration",
                    "precedingXVelocity",
                    "__presence__",
                ),
                10,
            ),
            "exid_paper": (
                (
                    "x",
                    "y",
                    "xVelocity",
                    "yVelocity",
                    "xAcceleration",
                    "yAcceleration",
                    "__presence__",
                ),
                7,
            ),
        }
        for experiment in (
            NGSIM_PAPER,
            HIGHD_PAPER,
            EXID_PAPER,
        ):
            with self.subTest(experiment=experiment.name):
                schema = get_dataset_schema(experiment.dataset)
                resolved = resolve_feature_spec(
                    schema,
                    experiment.feature_spec,
                )
                channels, input_dim = expected[experiment.name]
                self.assertEqual(resolved.tensor_channels, channels)
                self.assertEqual(resolved.input_dim, input_dim)
                self.assertEqual(resolved.position_indices, (0, 1))

    def test_semantic_group_may_resolve_to_multiple_channels(self):
        ngsim = get_dataset_schema("ngsim")
        highd = get_dataset_schema("highd")
        spec = FeatureSpec(groups=("position", "velocity"))
        self.assertEqual(
            len(resolve_feature_spec(ngsim, spec).raw_channels),
            3,
        )
        self.assertEqual(
            len(resolve_feature_spec(highd, spec).raw_channels),
            4,
        )

    def test_input_dim_changes_with_feature_selection(self):
        ngsim = get_dataset_schema("ngsim")
        canonical = resolve_feature_spec(
            ngsim,
            NGSIM_PAPER.feature_spec,
        )
        with_dimensions = resolve_feature_spec(
            ngsim,
            FeatureSpec(
                groups=(
                    "position",
                    "dimensions",
                    "velocity",
                    "acceleration",
                ),
                include_presence=True,
            ),
        )
        self.assertEqual(canonical.input_dim, 5)
        self.assertEqual(with_dimensions.input_dim, 7)
        self.assertNotEqual(canonical.input_dim, with_dimensions.input_dim)

    def test_commented_ablation_candidates_are_registered(self):
        self.assertIn(
            "space_headway",
            get_dataset_schema("ngsim").feature_groups,
        )
        self.assertIn(
            "time_to_collision",
            get_dataset_schema("highd").feature_groups,
        )
        self.assertIn(
            "lead_time_to_collision",
            get_dataset_schema("exid").feature_groups,
        )


if __name__ == "__main__":
    unittest.main()
