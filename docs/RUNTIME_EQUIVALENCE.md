# Runtime equivalence certification

The canonical `timeba.Timeba` extraction was certified against the untouched
historical full NGSIM implementation preserved at commit
`1114c0dd976b07914010827ddd26f4f43f9d70e6` on branch `orin`.

Certification infrastructure commit:

```text
3c1065042dc447e3c8f2c54609b617a77867e5e8
```

Certified canonical configuration:

- selected input channels: 5
- aligned history: 24
- prediction horizon: 50
- modes: 6

Results:

- 316 ordered state_dict entries were identical
- all state_dict keys and tensor shapes were identical
- total/trainable parameter count: 38,629,465
- historical to canonical `strict=True` loading succeeded
- canonical to historical `strict=True` loading succeeded
- actor encoder parity succeeded
- actor-to-actor interaction parity succeeded
- prediction-head parity succeeded
- classification maximum absolute error: 0.0
- trajectory-regression maximum absolute error: 0.0
- final certification: equivalent

Tested environment:

- Python 3.10.20
- PyTorch 2.1.2+cu121
- mamba-ssm 1.2.2
- causal-conv1d 1.2.2.post1
- NVIDIA L40

The one-time legacy-vs-canonical execution scripts are not required by the
cleaned main branch after the historical implementation is removed. The golden
state-dict manifest remains as a regression test, and the complete original
source remains permanently available in `orin`.

Verification scope:

- runtime implementation equivalence: verified
- synthetic pipeline consistency: verified
- historical NGSIM/highD/exiD data-transformation parity: verified
- complete paper-metric reproduction after cleanup: not rerun

Runtime certification establishes implementation equivalence for the extracted
model. It is not a rerun of full dataset training or every metric table in the
paper.
