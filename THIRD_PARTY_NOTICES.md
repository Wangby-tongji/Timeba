# Third-party and historical attribution

The canonical neural-network implementation in `timeba/models/` and the
historical loss in `timeba/losses.py` were extracted from research code carrying
Uber Technologies / LaneGCN notices. Those source-file notices are retained.
The extracted files also carry dated modification notices describing their
2026 modularization and integration. The repository `LICENSE` is unchanged.

Timeba depends on `mamba-ssm`, developed by Albert Gu, Tri Dao, and
contributors. The cleaned repository imports the installed dependency and no
longer vendors the historical `mamba_block.py` copy.

Historical files carrying Argo AI, Amazon, Albert Gu / Tri Dao, Uber, or other
notices are preserved without alteration on the `orin` branch. They are removed
from the cleaned branch only because they are not dependencies of the canonical
pipeline:

- the Argo AI forecasting CSV loader
- Amazon-derived time-feature utilities
- the vendored Mamba helper
- historical attention, dataset, visualization, and experiment utilities

Removing those unused copies from the cleaned branch does not remove their
copyright or license history from `orin`.
