# Timeba runtime equivalence certification

These checks certify the canonical `timeba.models.timeba.Timeba` model against
the unchanged historical `NGSIM24_5_4.Net` implementation.

Use the historical Timeba environment. Record its existing package versions;
do not upgrade `mamba_ssm` merely to run these checks.

From the repository root:

```bash
python -m tests.compatibility.runtime_manifest
python -m tests.compatibility.certify_runtime
python -m unittest tests.compatibility.test_timeba_compatibility -v
```

The certification runner writes:

- `runtime_legacy_manifest.json`: ordered runtime state-dict keys, tensor
  shapes, parameter/buffer classification, counts, and environment versions.
- `runtime_certification.json`: ordered-key, shape, parameter-count,
  bidirectional `strict=True`, final-output, and intermediate-stage results.

The runtime legacy manifest is the source of truth if it differs from the
source-derived expectation in `golden_manifest.py`. A source expectation
difference must be investigated as a dependency/version issue; it must not be
resolved by changing the model merely to make the expectation pass.

Forward certification requires CUDA because the historical implementation
unconditionally transfers inputs with `.cuda()`. The runner compares complete
classification and trajectory outputs with `rtol=1e-5`, `atol=1e-6`, and
records actor encoder, A2A, and prediction-head diagnostics.

The deterministic synthetic input uses `input_dim=5`, `history_len=24`,
`pred_len=50`, and `num_modes=6`.
