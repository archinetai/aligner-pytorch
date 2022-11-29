import torch
import math
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from einops import rearrange, reduce, repeat
from .utils import exists
from .mas import mas


@torch.no_grad()
def get_alignment_from_embeddings(
    x_embedding: Tensor,
    y_embedding: Tensor,
    x_mask: Optional[Tensor] = None,
) -> Tensor:
    _, ty, d = y_embedding.shape
    # Compute multivariate gaussian log PDF: log N(x|mu=y, Σ=I)
    const = -0.5 * math.log(2 * math.pi) * d
    factor = -0.5 * torch.ones(x_embedding.shape).to(x_embedding)
    y_sq = torch.einsum("b x d, b y d -> b x y", factor, y_embedding**2)
    y_mu = torch.einsum("b x d, b y d -> b x y", 2 * factor * x_embedding, y_embedding)
    x_sq = reduce(factor * x_embedding**2, "b tx d -> b tx 1", "sum")
    log_prior = y_sq - y_mu + x_sq + const
    # Mask xs if provided
    a_mask = repeat(x_mask, "b tx -> b tx ty", ty=ty) if exists(x_mask) else None
    # Compute MAS alignment
    alignment = mas(log_prior, mask=a_mask)
    return alignment


def get_sequential_masks(lengths: Tensor, length_max: Optional[int] = None) -> Tensor:
    if not exists(length_max):
        length_max = int(lengths.max().item())
    length_max = int(length_max)
    x = rearrange(torch.arange(length_max).to(lengths), "n -> 1 n")
    y = rearrange(lengths, "b -> b 1")
    return x < y


def get_alignment_from_duration(
    duration: Tensor,
    mask: Tensor,
) -> Tensor:
    b, tx, ty = mask.shape
    duration_cum = torch.cumsum(duration, dim=1)
    # Compute paths matrix filled with True on the lower diagonal
    paths = get_sequential_masks(
        lengths=rearrange(duration_cum, "b tx -> (b tx)"), length_max=ty
    )
    paths = rearrange(paths, "(b tx) ty -> b tx ty", b=b)
    # Get mask paths matrix to get only a single path by padding top and inverting
    paths_mask = ~F.pad(paths, pad=(0, 0, 1, 0))[:, :-1, :]
    # Get single path and mask unused
    paths = paths * paths_mask * mask
    return paths.long()


@torch.no_grad()
def get_alignment_from_duration_embedding(
    embedding: Tensor,  # [b, tx]
    scale: float = 1.0,
    mask: Optional[Tensor] = None,  # [b, tx]
    y_length: Optional[int] = None,
) -> Tensor:  # [b, tx, ty]
    b, tx, device = *embedding.shape, embedding.device
    # Default mask to all xs if not provided
    x_mask = mask if exists(mask) else torch.ones((b, tx), device=device).bool()
    assert x_mask.shape == embedding.shape, "mask must have same shape as embedding"
    # Get int duration by exponentiating and ceiling, then scaling by duration scale
    duration = torch.exp(embedding)
    duration = torch.ceil(duration) * scale
    duration = duration * x_mask
    # Compute total duration per item (clamp if below 1)
    duration_total = torch.clamp_min(reduce(duration, "b tx -> b", "sum"), 1).long()
    # Get max duration over all items
    duration_max = y_length if exists(y_length) else int(duration_total.max())
    # Get ys mask and attn matrix mask
    y_mask = get_sequential_masks(lengths=duration_total, length_max=duration_max)  # type: ignore # noqa
    a_mask = rearrange(x_mask, "b tx -> b tx 1") * rearrange(y_mask, "b ty -> b 1 ty")
    # Get masked attn paths from duration
    return get_alignment_from_duration(duration=duration, mask=a_mask)
