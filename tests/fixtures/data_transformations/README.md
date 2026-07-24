# Historical transformation golden fixtures

These small, weight-free fixtures were generated from the actual historical
loaders before those loaders were removed from the cleaned branch:

- NGSIM: `visualize_dev/data.py`
- highD: `datah.py`
- exiD: `datae.py`

Each loader processed the deterministic structured trajectories produced by
`tests.synthetic_data.make_synthetic_frame`. The arrays preserve the certified
channel order, actor order, history/future slicing, coordinate transform,
targets, masks, origins, and rotations. The source Stage 4 parity commit is
embedded in each `.npz` file.

The committed test reads these fixtures with `allow_pickle=False`; it does not
import or execute historical code. The untouched loader sources remain
available on the `orin` branch.
