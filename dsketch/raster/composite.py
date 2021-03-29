"""
Functions for compositing multiple rasters into a single image.
"""

import torch
from torch.nn.functional import softmax
from torch.types import Number


def softor(rasters: torch.Tensor, dim=1, keepdim=False) -> torch.Tensor:
    """Batched soft-or operation to combine rasters.

    Args:
      rasters: torch.Tensor of size [batch, *, nrasters, D*] representing stacks of nrasters images that will
               be composited into a single image
      dim: dimension over which to apply the composition
      keepdim: should the dimension be retained

    Returns:
      a torch.Tensor of size [batch, *, D*] representing the batch of composited images

    """
    rasters = 1 - rasters
    rasters = rasters.prod(dim=dim, keepdim=keepdim)  # product over nrasters dim
    return 1 - rasters


def over_recursive(rasters: torch.Tensor, dim=1) -> torch.Tensor:
    """Batched over operation to combine rasters.

    This is the straightforward recursive version and is very slow for multiple images (but fine for composing pairs).

    Args:
      rasters: torch.Tensor of size [batch, *, nrasters, D*] representing stacks of nrasters images that will
               be composited into a single image
      dim: dimension over which to apply the composition

    Returns:
      a torch.Tensor of size [batch, rows, cols] representing the batch of composited images.

    """
    output = torch.zeros(*rasters.shape[0:dim], *rasters.shape[dim+1:],
                         dtype=rasters.dtype, device=rasters.device)

    for i in range(rasters.shape[dim]):
        ith = rasters.select(dim, i)
        output = ith + output * (1.0 - ith)

    return output


def over(rasters: torch.Tensor, dim=1, keepdim=False) -> torch.Tensor:
    """Batched over operation to combine rasters.

    This is version implements the unwrapped version of over and is much faster. Note that the unwrapped version
    involves computing a visibility function that is the product of the previous alphas (not including the current one)
    for each image being composited. PyTorch's cumprod operation could be used to compute this, but it would include
    the current alpha. Instead we use a numerically stable approach with exp(cumsum(log(a0..ai)) - log(ai)) to compute
    this (the subtraction removes the current alpha).

    TODO: This almost certainly could be optimised with a custom CUDA kernel or Tensor Comprehension

    Args:
      rasters: torch.Tensor of size [batch, *, nrasters, D*] representing stacks of nrasters images that will
               be composited into a single image
      dim: dimension over which to apply the composition

    Returns:
      a torch.Tensor of size [batch, *, D*] representing the batch of composited images

    """
    rasters = rasters.clip(0, 1)

    linv = (1 - rasters) + 1e-10  # .clamp(0)
    linvrasters = linv.log()
    vis = (linvrasters.cumsum(dim=dim) - linvrasters).exp()
    comp = rasters * vis
    comp = comp.sum(dim=dim, keepdim=keepdim)
    # print("CLOSE", torch.isclose(comp, over_recursive(rasters)).all())
    return comp


def smoothmax(rasters: torch.Tensor, dim=1, temperature: Number = 1.) -> torch.Tensor:
    """Batched smoothmax operation to combine rasters.

    Args:
      rasters: torch.Tensor of size [batch, *, nrasters, D*] representing stacks of nrasters images that will
               be composited into a single image
      dim: dimension over which to apply the composition
      temperature: temperature for the smoothmax controlling to what extent the maximum takes precidence. (Default
                   value = 1.)

    Returns:
      a torch.Tensor of size [batch, *, D*] representing the batch of composited images

    """
    return (softmax(rasters / temperature, dim=1) * rasters).sum(dim=dim)
