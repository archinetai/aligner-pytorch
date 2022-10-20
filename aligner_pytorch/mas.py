from aligner_pytorch.mas_c import compute_mas_alignement
import torch
from torch import Tensor
from typing import Optional


def mas(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    b, m, n, device = *x.shape, x.device

    values = x.clone().to(dtype=torch.float32, device="cpu").numpy()
    paths = torch.zeros_like(x, dtype=torch.int32, device="cpu").numpy()

    ms = torch.tensor([m], dtype=torch.int32).repeat(b).numpy()
    ns = torch.tensor([n], dtype=torch.int32).repeat(b).numpy()

    compute_mas_alignement(paths, values, ms, ns)

    return torch.from_numpy(paths).to(device=device)
