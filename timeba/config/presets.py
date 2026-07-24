"""Paper-oriented and explicitly labelled historical experiment presets."""

from timeba.data.features import FeatureSpec

from .types import ExperimentConfig, TrainingConfig


NGSIM_PAPER = ExperimentConfig(
    name="ngsim_paper",
    dataset="ngsim",
    history_len=24,
    pred_len=50,
    feature_spec=FeatureSpec(
        groups=("position", "velocity", "acceleration"),
        include_presence=True,
    ),
    training=TrainingConfig(
        learning_rates=(1e-3, 1e-4, 4e-5),
        lr_milestones=(32, 42),
    ),
)

HIGHD_PAPER = ExperimentConfig(
    name="highd_paper",
    dataset="highd",
    history_len=72,
    pred_len=125,
    feature_spec=FeatureSpec(
        groups=(
            "position",
            "dimensions",
            "velocity",
            "acceleration",
            "preceding_velocity",
        ),
        include_presence=True,
    ),
    training=TrainingConfig(
        learning_rates=(1e-3, 1e-4, 1e-5),
        lr_milestones=(32, 42),
    ),
)

EXID_PAPER = ExperimentConfig(
    name="exid_paper",
    dataset="exid",
    history_len=72,
    pred_len=125,
    feature_spec=FeatureSpec(
        groups=("position", "velocity", "acceleration"),
        include_presence=True,
    ),
    training=TrainingConfig(
        learning_rates=(1e-3, 1e-4, 1e-5),
        lr_milestones=(32, 42),
    ),
)

_PRESETS = {
    config.name: config
    for config in (
        NGSIM_PAPER,
        HIGHD_PAPER,
        EXID_PAPER,
    )
}


def get_experiment_config(name):
    """Return a registered immutable experiment preset."""
    try:
        return _PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PRESETS))
        raise KeyError(f"unknown experiment {name!r}; choose from: {choices}") from exc
