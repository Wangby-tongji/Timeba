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

The historically grounded, paper-oriented presets currently resolve as
follows:

| Preset | Selected raw channels | Derived channel | Resolved input_dim |
| --- | --- | --- | ---: |
| NGSIM | `x, y, v_Vel, v_Acc` | presence | 5 |
| highD | `x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, precedingXVelocity` | presence | 10 |
| exiD | `x, y, xVelocity, yVelocity, xAcceleration, yAcceleration` | presence | 7 |

These numbers describe the current feature selections, not fixed properties of
NGSIM, highD, or exiD.

## Conservative temporal alignment

The manuscript describes a nominal 3-second observation horizon. The
historically recovered implementation uses aligned discrete history lengths of
24 steps for NGSIM and 72 steps for highD/exiD. These lengths preserve exact
temporal alignment through the three-stage downsampling hierarchy while
retaining the complete prediction horizon.

| Dataset | Nominal manuscript history | Historically recovered aligned history | Prediction horizon | Total window |
| --- | ---: | ---: | ---: | ---: |
| NGSIM | 30 | 24 | 50 | 74 |
| highD | 75 | 72 | 125 | 197 |
| exiD | 75 | 72 | 125 | 197 |

The aligned implementation lengths preserve the complete prediction horizon,
do not use more observation history than the nominal setting, satisfy all
three stride-2 temporal downsampling stages, and respect the available
sequence-window length. In particular, 24 NGSIM steps are an aligned discrete
implementation length; they are not described here as exactly 3 seconds.

The repository cleanup did not rerun the complete paper experiments, so the
implementation difference is documented rather than retrospectively resolved.
The repository does not infer that every reported experiment necessarily used
one recovered source snapshot.

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
