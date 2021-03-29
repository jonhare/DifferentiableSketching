from typing import Callable

import torch
from torch.nn.modules.loss import _Loss
from torch.types import Number

from ..raster.composite import softor
from ..raster.disttrans import line_edt2
from ..raster.raster import nearest_neighbour, compute_nearest_neighbour_sigma_bres


class Chamfer(_Loss):
    """Compute the chamfer loss between points, line segments, curves, etc.

    This chamfer distance implementation is designed to be flexible and can work with arbitrary data (as long as it can
    be represented as a tensor). The approach is as follows:
        1. the squared total distance transform is computed by applying the provided `dt2_fcn` to the input tensor.
        2. the total distance transform is computed by square-rooting the squared total distance transform.
        3. the target mask image is computed by applying the `ras_fcn` to the target tensor.
        4. the chamfer tensor is computed by computing the hadamard product of the total distance transform and target
           mask.
        5. the chamfer tensor is reduced using the specified reduction.

    Examples:
        - if the input represented line segments, then the `dt2_fcn` would need to compute the total (min across each
          segment) squared distance transform.
        - if the target represented line segments, then the `ras_fcn` would need to compute a binary raster of those
          segments (need not be differentiable)
        - if the target represented a binary target image, then the `ras_fcn` would just be None.

    Args:
        dt2_fcn: function to transform input into a batch of squared distance transform images
        ras_fcn: function to transform target into a batch of rasters; can be None if target is already a batch of
          rasters
        symmetric: if True, then the symmetric distance is computed (default False)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """

    def __init__(self, dt2_fcn: Callable[[torch.Tensor], torch.Tensor], ras_fcn: Callable[[torch.Tensor], torch.Tensor],
                 symmetric: bool = False, reduction: str = 'mean'):
        super(Chamfer, self).__init__(None, None, reduction)

        self.dt2_fcn = dt2_fcn
        self.ras_fcn = ras_fcn
        self.symmetric = symmetric

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return chamfer(input, target, self.dt2_fcn, self.ras_fcn, self.symmetric, self.reduction)


def chamfer(input: torch.Tensor, target: torch.Tensor, dt2_fcn: Callable[[torch.Tensor], torch.Tensor],
            ras_fcn: Callable[[torch.Tensor], torch.Tensor], symmetric: bool = False,
            reduction: str = 'mean') -> torch.Tensor:
    """
    Generic Chamfer distance function.

    This chamfer distance implementation is designed to be flexible and can work with arbitrary data (as long as it can
    be represented as a tensor). The approach is as follows:
        1. the squared total distance transform is computed by applying the provided `dt2_fcn` to the input tensor.
        2. the total distance transform is computed by square-rooting the squared total distance transform.
        3. the target mask image is computed by applying the `ras_fcn` to the target tensor.
        4. the chamfer tensor is computed by computing the hadamard product of the total distance transform and target
           mask.
        5. the chamfer tensor is reduced using the specified reduction.

    Examples:
        - if the input represented line segments, then the `dt2_fcn` would need to compute the total (min across each
          segment) squared distance transform.
        - if the target represented line segments, then the `ras_fcn` would need to compute a binary raster of those
          segments (need not be differentiable)
        - if the target represented a binary target image, then the `ras_fcn` would just be None.

    Args:
        input: batch of input data
        target: batch of target data
        dt2_fcn: function to transform input into a batch of squared distance transform images
        ras_fcn: function to transform target into a batch of rasters; can be None if target is already a batch of
          rasters
        symmetric: if True, then the symmetric distance is computed (default False)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Returns:
        Tensor containing the chamfer distance. For sum of mean reduction this will be scalar; for none it will be
        [batch, *] where * represents the shape of the grid.

    """
    if dt2_fcn is None:
        ic = input.sqrt()
    else:
        ic = dt2_fcn(input).sqrt()

    if ras_fcn is None:
        tc = target
    else:
        tc = ras_fcn(target)

    cham = ic * tc

    if symmetric:
        ic2 = ras_fcn(input)
        tc2 = dt2_fcn(target).sqrt()

        cham2 = ic2 * tc2
        cham += cham2

    if reduction == 'sum':
        return cham.sum()
    if reduction == 'mean':
        return cham.mean()

    return cham


def chamfer_line(input: torch.Tensor, target: torch.Tensor, grid: torch.Tensor, sigma: Number = None,
                 symmetric: bool = False, reduction: str = 'mean') -> torch.Tensor:
    """
    Chamfer distance for two sets of line segment parameters.

    Works by rasterising the target with nearest-neighbour to get a binary target `image' where 1-valued pixels are on
    the lines, and there are zeros elsewhere. Then the hardamard product of this binary target and the total distance
    transform of the lines in inputs is computed. The resulting tensor is then reduced according to the specified
    reduction.

    Args:
        input: the input line parameters [N, nlines_in, 4]
        target: the target line parameters [N, nlines_tgt, 4]
        grid: the rasterisation grid coordinates. For 1D data this will be shape [W, 1]; for 2D [H, W, 2],
          3D [D, H, W, 3].
        sigma: the threshold distance (Default value = None); if None it will be calculated to be equivalent to 1 px
          lines given by Bresenham's algorithm
        symmetric: if True, then the symmetric distance is computed (default False)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Returns:
        Tensor containing the chamfer distance. For sum of mean reduction this will be scalar; for none it will be
        [batch, *] where * represents the shape of the grid.

    """
    if sigma is None:
        sigma = compute_nearest_neighbour_sigma_bres(grid)

    def edt2f(inp):
        bs = inp.shape[0]
        edt2 = line_edt2(inp.view(-1, 4), grid).view(bs, -1, *grid.shape[0:-1])  # input is [N, n_lines, 4]
        edt2, _ = edt2.min(dim=1)  # take min across all EDTs to get the total EDT
        return edt2

    def rasf(inp):
        bs = inp.shape[0]
        edt2 = line_edt2(inp.view(-1, 4), grid).view(bs, -1, *grid.shape[0:-1])
        ras = nearest_neighbour(edt2, sigma)
        return softor(ras)

    return chamfer(input, target, edt2f, rasf, symmetric, reduction)
