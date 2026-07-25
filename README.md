# Timeba

Timeba is a multimodal vehicle-trajectory prediction model built around a
temporal U-shaped hierarchy and selective state-space (Mamba) blocks.

<p align="center">
  <img
    src="assets/Timeba_Encoder.jpg"
    alt="Timeba temporal U-shaped architecture"
    width="800"
  >
</p>

This branch provides one canonical model API:

```python
from timeba import Timeba

model = Timeba(
    input_dim=resolved_features.input_dim,
    pred_len=experiment.pred_len,
    num_modes=experiment.num_modes,
)
```

`input_dim` is derived from the selected feature groups. It is not an intrinsic
fixed dimension of NGSIM, highD, or exiD:

```text
DatasetSchema
  -> FeatureSpec
  -> ordered raw channels plus derived presence channel
  -> input_dim
```

The default presets currently resolve to 5 channels for NGSIM, 10 for highD,
and 7 for exiD. These values describe the present selections only. In
particular, the repository does not claim that the highD selection is proven to
be the exact feature configuration of every paper table.

## Implementation sequence lengths

The paper describes a nominal 3-second observation setting. The released
implementation uses temporally aligned discrete history lengths subject to the
temporal hierarchy and available sequence windows. See
[data and sequence configuration](docs/data-and-sequence-configuration.md) for
details.

## Installation

The certified environment used Python 3.10.20, PyTorch 2.1.2+cu121,
mamba-ssm 1.2.2, causal-conv1d 1.2.2.post1, and an NVIDIA L40. Other compatible
versions may also work. See [installation](docs/INSTALL.md).

## Prepared data

The official pipeline consumes prepared per-scene CSV files:

```text
DATA_ROOT/
  train/*.csv
  val/*.csv
  test/*.csv
```

See [prepared input format](docs/DATA_FORMAT.md). The cleanup did not verify a
complete raw-download-to-training pipeline for every dataset, so unverified
historical preprocessing scripts are not presented as official tooling.

## Training and evaluation

Use the single config-driven CLI:

```bash
python -m scripts.train \
  --experiment ngsim_paper \
  --data-root /path/to/prepared/data \
  --output-dir /path/to/new/run
```

Resume a canonical checkpoint with the same config-driven entrypoint; existing
metric records in the output directory are validated and preserved:

```bash
python -m scripts.train \
  --experiment ngsim_paper \
  --data-root /path/to/prepared/data \
  --output-dir /path/to/existing/run \
  --resume /path/to/existing/run/epoch_010.ckpt
```

```bash
python -m scripts.evaluate \
  --experiment ngsim_paper \
  --data-root /path/to/prepared/data \
  --checkpoint /path/to/checkpoint.ckpt \
  --split val \
  --output /path/to/metrics.json
```

The repository provides historically grounded, paper-oriented presets named
`ngsim_paper`, `highd_paper`, and `exid_paper`. Their feature selections remain
configurable. Checkpoint loading is strict by default, and evaluation uses each
preset's configured prediction horizon while preserving K=1/K=6 semantics.

## Validation status

The canonical model extraction passed runtime-equivalence certification against
the historical NGSIM implementation:

- 316 identical ordered state-dict entries
- 38,629,465 parameters
- bidirectional `strict=True` checkpoint loading
- actor encoder, actor interaction, and prediction-head parity
- classification and trajectory maximum absolute error of 0.0

See [runtime equivalence](docs/RUNTIME_EQUIVALENCE.md).

Synthetic tests additionally cover dataset transformation, collation, forward,
historical loss, backward, Adam update, checkpoint save/reload, and evaluation
shapes for the NGSIM, highD, and exiD presets. These tests establish code-chain
consistency; they are not a full retraining or numerical reproduction of every
paper table. Full paper metrics were not rerun during repository cleanup.

## Attribution and citation

The repository `LICENSE` remains unchanged. Relevant source-file notices are
preserved. See [third-party notices](THIRD_PARTY_NOTICES.md).

If you use Timeba in research, cite the Timeba paper and the Mamba work used by
the implementation.
