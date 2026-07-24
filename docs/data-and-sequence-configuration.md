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

The paper-oriented presets currently resolve as follows:

| Preset | Selected raw channels | Derived channel | Resolved input_dim |
| --- | --- | --- | ---: |
| NGSIM | `x, y, v_Vel, v_Acc` | presence | 5 |
| highD | `x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, precedingXVelocity` | presence | 10 |
| exiD | `x, y, xVelocity, yVelocity, xAcceleration, yAcceleration` | presence | 7 |

These numbers describe the current feature selections, not fixed properties of
NGSIM, highD, or exiD.

## Conservative temporal alignment

The implementation uses aligned histories that do not exceed the nominal paper
histories, preserve the required prediction horizons, fit within the available
trajectory windows, and remain compatible with the three temporal
downsampling/upsampling stages:

| Dataset | Nominal history | Canonical aligned history | Prediction | Required window |
| --- | ---: | ---: | ---: | ---: |
| NGSIM | 30 | 24 | 50 | 74 |
| highD | 75 | 72 | 125 | 197 |
| exiD | 75 | 72 | 125 | 197 |

The aligned values are intentional implementation policy. They must not be
changed solely to reproduce nominal prose values when that would violate
temporal U-shape alignment or the available trajectory window.

Synthetic parity tests compare these canonical transformations with the
historical loaders. They validate channel ordering, slicing, coordinate
transforms, targets, masks, and actor metadata without redesigning historical
velocity or acceleration handling.
