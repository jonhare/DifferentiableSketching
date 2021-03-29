"""
Functions for rasterising based on the image formed from the squared distance transform of the object being rasterised.
"""

import torch
from torch.types import Number


def compute_nearest_neighbour_sigma(grid: torch.Tensor) -> Number:
    """Compute the sigma2 value required for `nearest_neighbour` to rasterise objects exactly one pixel wide.

    This version computes sigma2 such that any pixel the line passes through will be selected.

    Args:
        grid: the rasterisation grid coordinates. For 1D data this will be shape [W, 1]; for 2D [H, W, 2],
          3D [D, H, W, 3].

    Returns:
        the value of sigma2 to use in `nearest_neighbour`
    """
    dim = grid.shape[-1]
    delta2 = (grid[(1,)*dim] - grid[(0,)*dim]) ** 2
    return delta2.sum().sqrt() / 2


def compute_nearest_neighbour_sigma_bres(grid: torch.Tensor) -> Number:
    """Compute the sigma2 value required for `nearest_neighbour` to rasterise objects exactly one pixel wide.

    This version computes sigma2 such that the resultant raster looks like what you would get using Bresenham's
    algorithm (pixels for which the line just passes [at an angle] are not selected).

    Args:
        grid: the rasterisation grid coordinates. For 1D data this will be shape [W, 1]; for 2D [H, W, 2],
          3D [D, H, W, 3]. The grid sampling is assumed to be square!

    Returns:
        the value of sigma2 to use in `nearest_neighbour`
    """
    dim = grid.shape[-1]
    delta = (grid[(1,)*dim] - grid[(0,)*dim]).abs() / 2
    return delta[0]


def nearest_neighbour(dt2: torch.Tensor, sigma2: Number = 1) -> torch.Tensor:
    """Nearest-neighbour rasterisation function.
    
    Sets pixels in the distance transform to 1 if they are less than equal to sigma**2
    and zero otherwise. Note this doesn't have usable gradients!

    Args:
      dt2: the squared distance transform
      sigma2: the threshold distance (Default value = 1)

    Returns:
      the rasterised image

    """
    return (dt2 <= sigma2 ** 2) * 1.


def sigmoid(dt2: torch.Tensor, sigma2: Number) -> torch.Tensor:
    """Sigmoidal rasterisation function.
    
    Computes $$2 \\times \\sigmoid(-dt^2 / \\sigma)$$ giving values near 1.0
    for points in the distance transform with near-zero distance, and falling
    off as distance increases.

    Args:
      dt2: the squared distance transform
      sigma2: the rate of fall-off. Larger values result in greater line width,
              but also larger gradient flow across the raster

    Returns:
      the rasterised image

    """
    return torch.sigmoid(-1 * dt2 / sigma2) * 2.


def exp(dt2: torch.Tensor, sigma2: Number) -> torch.Tensor:
    """Exponentiated rasterisation function.
    
    Computes $$\\exp(-dt^2 / \\sigma)$$ giving values near 1.0 for points in
    the distance transform with near-zero distance, and falling off as distance
    increases.

    Args:
      dt2: the squared distance transform
      sigma2: the rate of fall-off. Larger values result in greater line width,
              but also larger gradient flow across the raster

    Returns:
      the rasterised image

    """
    return torch.exp(-1 * dt2 / sigma2)
