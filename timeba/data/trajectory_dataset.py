"""Shared trajectory construction for NGSIM, highD, exiD, and inD."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .features import resolve_feature_spec


class TrajectoryDataset(Dataset):
    """Build historical Timeba scene dictionaries from CSV-like samples."""

    def __init__(self, samples, schema, experiment, train=False):
        self.samples = tuple(samples)
        self.schema = schema
        self.experiment = experiment
        self.train = bool(train)
        self.resolved_features = resolve_feature_spec(
            schema,
            experiment.feature_spec,
        )

    @classmethod
    def from_directory(cls, directory, schema, experiment, train=False):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset split directory not found: {directory}")
        samples = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix == ".csv"
        )
        if not samples:
            raise ValueError(f"no CSV samples found in {directory}")
        return cls(samples, schema, experiment, train=train)

    def __len__(self):
        return len(self.samples)

    def _read_sample(self, index):
        sample = self.samples[index]
        if isinstance(sample, pd.DataFrame):
            frame = copy.deepcopy(sample)
        else:
            frame = pd.read_csv(Path(sample))
        return frame.reset_index(drop=True)

    def _validate_columns(self, frame):
        required = set(self.schema.identity_columns)
        required.update(self.resolved_features.raw_channels)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(
                f"{self.schema.name} sample is missing required columns: {missing}"
            )

    def _trajectory_records(self, frame):
        schema = self.schema
        unique_frames = np.sort(frame[schema.frame_column].unique())
        if len(unique_frames) < self.experiment.total_len:
            raise ValueError(
                f"sample contains {len(unique_frames)} frames, "
                f"but {self.experiment.name} requires "
                f"{self.experiment.total_len}"
            )
        frame_to_step = {
            frame_value: index for index, frame_value in enumerate(unique_frames)
        }
        steps = frame[schema.frame_column].map(frame_to_step).to_numpy(np.int64)
        values = frame.loc[
            :,
            self.resolved_features.raw_channels,
        ].to_numpy(np.float32)

        grouped = frame.groupby(
            [schema.actor_id_column, schema.actor_type_column],
            sort=True,
        ).groups
        agent_keys = [
            key for key in grouped if key[1] == schema.agent_label
        ]
        if len(agent_keys) != 1:
            raise ValueError(
                f"expected exactly one {schema.agent_label!r} actor, "
                f"found {len(agent_keys)}"
            )

        ordered_keys = agent_keys + [key for key in grouped if key not in agent_keys]
        records = []
        for key in ordered_keys:
            indices = np.asarray(grouped[key], dtype=np.int64)
            actor_steps = steps[indices]
            order = actor_steps.argsort()
            records.append((values[indices][order], actor_steps[order]))
        return records

    @staticmethod
    def _contiguous_history_start(observed_steps, current_step):
        available = set(int(step) for step in observed_steps)
        start = current_step
        while start - 1 in available:
            start -= 1
        return start

    def __getitem__(self, index):
        frame = self._read_sample(index)
        self._validate_columns(frame)
        records = self._trajectory_records(frame)

        history_len = self.experiment.history_len
        pred_len = self.experiment.pred_len
        current_step = history_len - 1
        position_indices = self.resolved_features.position_indices

        agent_values, agent_steps = records[0]
        agent_current = np.flatnonzero(agent_steps == current_step)
        agent_previous = np.flatnonzero(agent_steps == current_step - 1)
        if len(agent_current) != 1 or len(agent_previous) != 1:
            raise ValueError(
                "AGENT must contain the final two historical trajectory steps"
            )

        orig = agent_values[agent_current[0], list(position_indices)].astype(
            np.float32,
            copy=True,
        )
        if self.train and self.experiment.rotation_augmentation:
            theta = float(np.random.rand() * np.pi * 2.0)
        else:
            previous = agent_values[
                agent_previous[0],
                list(position_indices),
            ]
            delta = previous - orig
            theta = float(np.pi - np.arctan2(delta[1], delta[0]))
        rot = np.asarray(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        )

        feats = []
        past = []
        ctrs = []
        gt_preds = []
        has_preds = []

        for values, steps in records:
            if current_step not in steps:
                continue

            gt_pred = np.zeros((pred_len, 2), dtype=np.float32)
            has_pred = np.zeros(pred_len, dtype=bool)
            future_mask = (steps >= history_len) & (
                steps < history_len + pred_len
            )
            future_steps = steps[future_mask] - history_len
            gt_pred[future_steps] = values[
                future_mask
            ][:, list(position_indices)]
            has_pred[future_steps] = True

            observed_mask = steps < history_len
            observed_steps = steps[observed_mask]
            observed_values = values[observed_mask]
            start = self._contiguous_history_start(
                observed_steps,
                current_step,
            )
            suffix_mask = observed_steps >= start
            observed_steps = observed_steps[suffix_mask]
            observed_values = observed_values[suffix_mask]

            feat = np.zeros(
                (history_len, self.resolved_features.input_dim),
                dtype=np.float32,
            )
            positions = observed_values[:, list(position_indices)]
            feat[observed_steps, :2] = (rot @ (positions - orig).T).T
            non_position_indices = [
                channel
                for channel in range(len(self.resolved_features.raw_channels))
                if channel not in position_indices
            ]
            if non_position_indices:
                feat[
                    np.ix_(observed_steps, non_position_indices)
                ] = observed_values[:, non_position_indices]
            if self.resolved_features.include_presence:
                feat[observed_steps, -1] = 1.0

            past_feat = np.zeros((history_len, 2), dtype=np.float32)
            past_feat[observed_steps] = positions

            x_min, x_max, y_min, y_max = self.experiment.pred_range
            current_position = feat[current_step, :2]
            if not (
                x_min <= current_position[0] <= x_max
                and y_min <= current_position[1] <= y_max
            ):
                continue

            ctrs.append(current_position.copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[observed_steps[0], :2] = 0.0
            feats.append(feat)
            past.append(past_feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        if not feats:
            raise ValueError("sample contains no actors at the current history step")
        if not has_preds[0].all():
            raise ValueError(
                "AGENT must contain the complete configured prediction horizon"
            )

        return {
            "city": self.schema.name,
            "feats": np.asarray(feats, dtype=np.float32),
            "past": np.asarray(past, dtype=np.float32),
            "ctrs": np.asarray(ctrs, dtype=np.float32),
            "orig": orig,
            "theta": theta,
            "rot": rot,
            "gt_preds": np.asarray(gt_preds, dtype=np.float32),
            "has_preds": np.asarray(has_preds, dtype=bool),
            "idx": index,
        }
