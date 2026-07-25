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

The released configurations use temporally aligned history lengths that match
the three-stage downsampling hierarchy while preserving the complete prediction
horizon and respecting the available trajectory window.

- **NGSIM:** 24 history steps and 50 prediction steps.
- **highD:** 72 history steps and 125 prediction steps.
- **exiD:** 72 history steps and 125 prediction steps.

These are the discrete sequence lengths used by the cleaned implementation.
They are documented here as implementation settings rather than as exact
conversions of the nominal observation duration stated in the manuscript.

The repository cleanup did not rerun the complete paper experiments, so no
claim is made that every reported experiment used one recovered source
snapshot.

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
