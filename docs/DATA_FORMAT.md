# Prepared trajectory input format

The cleaned repository consumes prepared trajectory scenes. It does not claim
that the retained code reproduces every dataset's complete raw-download to
training-split process.

`scripts/train.py` and `scripts/evaluate.py` expect:

```text
DATA_ROOT/
  train/*.csv
  val/*.csv
  test/*.csv
```

Each CSV represents one scene and must contain:

- `frame`: sortable frame identifier
- `id`: actor identifier
- `type`: exactly one actor labelled `AGENT`; remaining actors may be `OTHERS`
- every raw column selected by the experiment's `FeatureSpec`

The canonical paper-oriented presets require complete target windows:

| Preset | Required scene frames | History | Future |
| --- | ---: | ---: | ---: |
| NGSIM | 74 | 24 | 50 |
| highD | 197 | 72 | 125 |
| exiD | 197 | 72 | 125 |

Required selected columns are resolved from `DatasetSchema` and `FeatureSpec`;
they are not represented by a fixed intrinsic dataset dimension. See
`docs/CONFIGURATION.md`.

Historical preprocessing utilities were removed because the complete raw-data
pipeline and unit conventions could not be verified without the original
datasets. In particular, the historical NGSIM script contained dataset-specific
unit conversion and destructive output-directory behavior. Users should
prepare CSV scenes consistently with their source dataset and record any unit
conversion applied.
