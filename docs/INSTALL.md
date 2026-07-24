# Installation

Timeba requires PyTorch and the CUDA-backed Mamba operators for canonical model
execution. Install PyTorch first, using the official package source appropriate
for the host CUDA driver, and then install Mamba and the remaining Python
dependencies.

The runtime-equivalence and synthetic integration certifications used:

- Python 3.10.20
- PyTorch 2.1.2+cu121
- mamba-ssm 1.2.2
- causal-conv1d 1.2.2.post1
- NVIDIA L40

One compatible environment is:

```bash
conda create -n timeba python=3.10 -y
conda activate timeba

python -m pip install \
  torch==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  causal-conv1d==1.2.2.post1 \
  mamba-ssm==1.2.2

python -m pip install -r requirements.txt
```

Mamba and causal-conv1d wheels must be compatible with the installed PyTorch,
CUDA runtime, Python version, and C++ ABI. Avoid silently falling back to a
source build on machines without a matching CUDA compiler.

The certified versions are a tested compatibility point, not a declaration
that no other versions work.
