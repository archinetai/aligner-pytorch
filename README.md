Sequence alignement methods with helpers for PyTorch. 

## Install

```bash
pip install aligner-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/aligner-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/aligner-pytorch/)


## Usage

### MAS

MAS (Monotonic Alignment Search) from GlowTTS. This can be used to get the alignment of any (similarity) matrix. Implementation in optimized Cython. 

```py
from aligner_pytorch import mas 

sim = torch.rand(1, 4, 6) # [batch_size, m_rows, n_cols]
alignment = mas(x)

"""
sim = tensor([[
    [0.2, 0.8, 0.9, 0.9, 0.9, 0.4],
    [0.6, 0.8, 0.9, 0.7, 0.1, 0.4],
    [1.0, 0.4, 0.4, 0.2, 1.0, 0.7],
    [0.1, 0.3, 0.1, 0.7, 0.6, 0.9]
]])

alignment = tensor([[
    [1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
]], dtype=torch.int32)
"""
```

## Citations

Monotonic Alignment Search 
```bibtex
@misc{2005.11129,
Author = {Jaehyeon Kim and Sungwon Kim and Jungil Kong and Sungroh Yoon},
Title = {Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search},
Year = {2020},
Eprint = {arXiv:2005.11129},
}
```