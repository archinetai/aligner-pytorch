
# Aligner - PyTorch

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

sim = torch.rand(1, 4, 6) # [batch_size, x_length, y_length]
alignment = mas(sim)

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

### XY Embedding to Alignment
Used during training to get the alignement of a `x_embedding` with `y_embedding`, computes the log probability from a normal distribution and the alignment with MAS.
```py
from aligner_pytorch import get_alignment_from_embeddings

x_embedding = torch.randn(1, 4, 10)
y_embedding = torch.randn(1, 6, 10)

alignment = get_alignment_from_embeddings(
    x_embedding=torch.randn(1, 4, 10),  # [batch_size, x_length, features]
    y_embedding=torch.randn(1, 6, 10),  # [batch_size, y_length, features]
)                                       # [batch_size, x_length, y_length]

"""
alignment = tensor([[
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1]
]], dtype=torch.int32)
"""
```

### Duration Embedding to Alignment
Used during inference to compute the alignment from a trained duration embedding.
```py
from aligner_pytorch import get_alignment_from_duration_embedding

alignment = get_alignment_from_duration_embedding(
    embedding=torch.randn(1, 5),    # Embedding: [batch_size, x_length]
    scale=1.0,                      # Duration scale
    y_length=10                     # (Optional) fixes maximum output y_length
)                                   # Output alignment [batch_size, x_length, y_length]

"""
alignment  = tensor([[
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
]])
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
