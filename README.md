# Timeba: Temporal U-Shape Hierarchy with Selective State-Space Modeling for Trajectory Prediction

Timeba is a trajectory forecasting research codebase built on a LaneGCN-style pipeline, with **Mamba (Selective SSM)** blocks integrated into the actor encoder / 1D U-Net style temporal modeling.

> ⚠️ Note: This repo currently contains multiple experimental model variants (e.g., `NGSIM24_5_4.py`, `highD*.py`, `inD*.py`, `exiD*.py`) and training scripts (`train1.py`, `train2.py`). The default data pipeline in this repo is based on **Argoverse Motion Forecasting (v1.1)**.

---

## Table of Contents
- [Requirements](#requirements)
- [Install (Recommended)](#install-recommended)
- [Mamba Installation Notes](#mamba-installation-notes)
- [Prepare Data (Argoverse Forecasting v1.1)](#prepare-data-argoverse-forecasting-v11)
- [Training](#training)
- [Testing / Evaluation](#testing--evaluation)
- [Troubleshooting](#troubleshooting)
- [License & Citation](#license--citation)

---

## Requirements

### OS / Hardware
- **Linux + NVIDIA GPU** is strongly recommended.
- `mamba-ssm` / `causal-conv1d` require **CUDA ≥ 11.6** and **PyTorch ≥ 1.12** (GPU build).  
  See official Mamba installation requirements:
  - https://github.com/state-spaces/mamba

> If you only have CPU, you will need to disable Mamba usage or refactor the code.

### Python
- ✅ Recommended: **Python 3.10** (3.11 also works in most cases)
- Not recommended: Python 3.7/3.8 (too old for modern PyTorch + mamba ecosystem)

---

## Install (Recommended)

### 1) Create a clean environment
Using conda:

```bash
conda create -n timeba python=3.10 -y
conda activate timeba
python -V
```

### 2) Install PyTorch (GPU)
Install a CUDA-enabled PyTorch build that matches your driver / CUDA runtime. Choose **one**:

**Option A (conda, recommended if you use conda):**
```bash
# Example (CUDA 12.1)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Option B (pip):**
Use the PyTorch official install selector and install the correct CUDA wheel for your system.

### 3) Install core python deps
```bash
pip install -U pip setuptools wheel
pip install numpy scipy pandas tqdm ipdb matplotlib scikit-image
```

### 4) Install Mamba (mamba-ssm + causal-conv1d)
**Recommended one-liner (installs both):**
```bash
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
```

If you prefer explicit installation:
```bash
pip install "causal-conv1d>=1.4.0" --no-build-isolation
pip install "mamba-ssm" --no-build-isolation
```

### 5) Sanity check
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())

from mamba_ssm import Mamba
x = torch.randn(2, 64, 16, device="cuda")
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
y = m(x)
print("mamba ok:", y.shape)
PY
```

---

## Mamba Installation Notes

- Official Mamba install guide suggests:
  - `pip install mamba-ssm[causal-conv1d]`
  - add `--no-build-isolation` if pip complains about PyTorch/CUDA toolchain.
- Requirements (official):
  - Linux, NVIDIA GPU, CUDA ≥ 11.6, PyTorch ≥ 1.12.

If installation fails:
1. **Make sure PyTorch is installed first** (GPU build), then install `mamba-ssm`.
2. Try verbose install:
   ```bash
   pip install "mamba-ssm[causal-conv1d]" --no-build-isolation -v
   ```
3. Ensure `nvcc -V` works if pip is building from source.
4. Prefer matching versions: (PyTorch CUDA wheel) ↔ (system driver).

---

## Prepare Data

A helper script is provided:

```bash
bash get_data.sh
```

---

## Training

### Single GPU training
Use `train.py` (prints parameter count and logs into `save_dir/log`):

```bash
python train.py -m NGSIM24_5_4
# or
python train.py -m highD64
# or
python train.py -m inD3_5_6
```

Resume / eval:

```bash
python train.py -m NGSIM24_5_4 --resume /absolute/path/to/ckpt.pth
python train.py -m NGSIM24_5_4 --eval --weight /absolute/path/to/ckpt.pth
```

### Multi-GPU / Distributed
This repo historically included Horovod instructions, but your current training entrypoints may differ by branch/version.
If you decide to use Horovod, install `mpi4py` + `horovod` and launch with `horovodrun`.

---

## Testing / Evaluation

### Run inference on val/test
```bash
python test.py -m NGSIM24_5_4 --weight /home/ps/WorkSpaces/wby/Timeba/Timeba/results/NGSIM24_5_4/48.000.ckpt --split=val
python test.py -m NGSIM24_5_4 --weight /home/ps/WorkSpaces/wby/Timeba/Timeba/results/NGSIM24_5_4/48.000.ckpt --split=test
```

For validation split, the script calls Argoverse metric computation.

---

## Troubleshooting

### 1) `mamba-ssm / causal-conv1d` install fails
- Ensure CUDA-enabled PyTorch is installed first.
- Retry with:
  ```bash
  pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
  ```
- Check CUDA toolchain / driver compatibility.

### 2) Shape mismatch in temporal U-Net
Some model variants downsample time by stride=2 multiple times.
Ensure your input sequence length is compatible (often needs to be divisible by 8).

---

## License & Citation

- This repository contains research code. Please check `LICENSE` if provided.
- If you use Mamba, please cite the Mamba paper (see the official Mamba repo / PyPI page).
