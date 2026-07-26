# Data features and aligned sequence lengths

Timeba does not assign an immutable input dimension to a dataset. The canonical
configuration resolves model input channels in four explicit steps:

```text
DatasetSchema
  -> selected FeatureSpec
  -> ordered raw tensor channels plus derived channels
  -> input_dim
```

A semantic feature group may expand to one or more tensor channels. For
example, the NGSIM velocity group currently expands to `v_Vel`, whereas the
highD and exiD velocity groups expand to `xVelocity, yVelocity`. The presence
mask is a derived channel rather than a raw dataset attribute. Consequently,
changing a `FeatureSpec` can change `input_dim` without changing the dataset
schema.

The default dataset configurations currently resolve as follows:

| Preset | Selected raw channels | Derived channel | Resolved input_dim |
| --- | --- | --- | ---: |
| NGSIM | `x, y, v_Vel, v_Acc` | presence | 5 |
| highD | `x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, precedingXVelocity` | presence | 10 |
| exiD | `x, y, xVelocity, yVelocity, xAcceleration, yAcceleration` | presence | 7 |

These numbers describe the current feature selections, not fixed properties of
NGSIM, highD, or exiD.

## Sequence configuration

Timeba uses three stride-2 temporal downsampling stages. To maintain exact
temporal alignment throughout the U-shaped hierarchy, the default
configurations use the following sequence lengths:

- **NGSIM:** 24 history steps and 50 prediction steps.
- **highD:** 72 history steps and 125 prediction steps.
- **exiD:** 72 history steps and 125 prediction steps.

These settings preserve the complete prediction horizon while remaining
compatible with the temporal hierarchy and the available trajectory windows.
The manuscript describes the observation horizon in seconds, whereas the
implementation uses dataset-dependent discrete sequence lengths that
approximate this horizon while satisfying temporal-alignment constraints.

## Historically preserved learning-rate schedules

The cleaned NGSIM preset retains the historical source schedule ending at
`4e-5`. The cleaned highD and exiD presets preserve the final learning rate of
`1e-5` found in their corresponding historical source snapshots, with
milestones at epochs 32 and 42.

The manuscript prose reports a different final learning rate of `4e-5` for
those settings. Because complete metric reproduction was not rerun during
repository cleanup, this difference is documented rather than silently
changed. No claim is made here about which value produced a published result.

Synthetic parity tests compare these canonical transformations with small
golden outputs generated from the actual historical loaders before cleanup.
They validate channel ordering, slicing, coordinate transforms, targets, masks,
and actor metadata without importing historical code or redesigning historical
velocity or acceleration handling.
