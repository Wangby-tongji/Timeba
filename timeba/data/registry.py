"""Dataset schema registry and canonical dataset construction."""

from .schemas import DATASET_SCHEMAS
from .trajectory_dataset import TrajectoryDataset


def get_dataset_schema(name):
    try:
        return DATASET_SCHEMAS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_SCHEMAS))
        raise KeyError(f"unknown dataset {name!r}; choose from: {choices}") from exc


def build_dataset(samples, experiment, train=False):
    schema = get_dataset_schema(experiment.dataset)
    return TrajectoryDataset(
        samples=samples,
        schema=schema,
        experiment=experiment,
        train=train,
    )
