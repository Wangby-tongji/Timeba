# Canonical checkpoint manifest regression

Stage 2B certified `timeba.Timeba` against the unchanged historical
`NGSIM24_5_4.Net` implementation on the `orin` branch. The one-time
legacy-vs-canonical runner was removed with the duplicated historical source.
Its results and tested environment are recorded in
`docs/RUNTIME_EQUIVALENCE.md`.

`golden_manifest.py` retains the certified ordered state-dict contract without
model weights. `test_canonical_manifest.py` checks:

- all 316 ordered state-dict entries
- every tensor shape
- the 38,629,465 parameter count
- checkpoint-sensitive registered-but-unused parameters
- strict loading between fresh canonical instances

Run:

```bash
python -m unittest tests.compatibility.test_canonical_manifest -v
```

The manifest configuration (`input_dim=5`, aligned history 24, prediction 50,
six modes) is the certified historical NGSIM fixture. It is not an intrinsic
dimension declaration for the NGSIM dataset.
