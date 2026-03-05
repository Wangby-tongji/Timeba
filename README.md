# README

## Architecture

![Timeba Temporal U-Shape Hierarchy](./assets/timeba_architecture.png)

The model features an explicit 1D temporal U-shape hierarchy with selective state-space (Mamba/SSM) blocks. The architecture consists of multiple Timeba blocks at different temporal resolutions (R, N, L at the top level, and progressively finer resolutions B, D4, L/2 and B, D2, L/4), with skip connections for information flow and Conv1d operations for feature transformation.

---