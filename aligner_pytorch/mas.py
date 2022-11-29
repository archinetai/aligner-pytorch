from aligner_pytorch.mas_c import compute_mas_alignement
import torch
from torch import Tensor
from typing import Optional
from .utils import exists


def mas(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    device = x.device

    values = x.detach().clone().to(dtype=torch.float32, device="cpu").numpy()
    paths = torch.zeros_like(x, dtype=torch.int32, device="cpu").numpy()

    mask = mask.clone() if exists(mask) else torch.ones_like(x)
    mask = mask.to(dtype=torch.int32, device="cpu").numpy()

    # ms = reduce(mask, 'b m n -> b m', 'sum')[:, 0]
    # ns = reduce(mask, 'b m n -> b n', 'sum')[:, 0]

    b, m, n = x.shape
    ms = torch.tensor([m], dtype=torch.int32).repeat(b).numpy()
    ns = torch.tensor([n], dtype=torch.int32).repeat(b).numpy()

    compute_mas_alignement(paths, values, ms, ns)

    return torch.from_numpy(paths).to(device=device)
