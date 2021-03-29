from typing import Union, Sequence, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.types import _int, Number

from ..utils import GaussianScaleSpace


def _fake_pyr(target: torch.Tensor, octaves: _int, intervals: _int, dim: _int, downsample: bool):
    # Make a 'fake' pyramid/ss dimensionally compatible with the gaussian one, but without doing
    # any blurring
    if not downsample:
        return [target] * (octaves * intervals)

    result = [target]
    for o in range(octaves):
        result.extend([result[-1]] * intervals)

        if dim == 1:
            result[-1] = result[-1][..., 0::2]
        if dim == 2:
            result[-1] = result[-1][..., 0::2, 0::2]
        if dim == 3:
            result[-1] = result[-1][..., 0::2, 0::2, 0::2]

    return result


class PyramidMSE(_Loss):
    """Mean Squared Error loss over a scale space or pyramid.

    This loss function is basically mean-squared error, but with a built-in multi-scale gaussian receptive field to
    allow surrounding pixels to contribute to the loss of an individual pixel. This is achieved by increasingly blurring
    the input (and optionally the target) with a Gaussian kernel (applied independently to each channel) to build a
    scale-space or pyramid representation. The MSE is then computed between the input and target at each the pyramid and
    the losses aggregated according to the specified reduction.

    This loss could for example be useful in the case where a reconstruction error of images is required, but the input
    is constrained to be near black and white whilst the target is greyscale; with the blurred loss the perceptual
    effect of patterns of black and white pixels creating percieved grey levels would be captured.

    Args:
        channels: Number of channels of the input tensors. Output will have this number of channels as well.
        octaves: Number of octaves in the pyramid. Each octave represents a doubling of sigma.
        intervals: Number of intervals to break each octave into
        init_sigma: Standard deviation blurring of the input images. Default of 0.5 is the minimum that would be
            possible without aliasing
        dim: number of dimensions for the gaussian
        downsample: if true then each octave is subsampled by taking every other pixel
        weight: if False, then each scale is weighted according to its size when computing the mean; if true then each
            time downsampling is used, the means of those downsampled layers is multiplied by 4^s before summation so
            that the number of pixels is equal in each scale. Can also be a tensor or sequence of length
            octaves*intervals+1 to specify a particular weight for each corresponding scale. Has no effect if reduction
            is 'none'.
        symmetric: if False, only the input is blurred. If True, then the target is also blurred.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
        - Target: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
    """

    def __init__(self, channels: _int, octaves: _int = 3, intervals: _int = 1, init_sigma: Number = 0.5, dim: _int = 2,
                 downsample=False, weight: Union[bool, Sequence[Number], torch.Tensor] = False, symmetric: bool = True,
                 reduction: str = 'mean'):
        super(PyramidMSE, self).__init__(None, None, reduction)

        self.pyr = GaussianScaleSpace(channels, octaves, intervals, init_sigma, dim, downsample)
        self.symmetric = symmetric

        if weight is False:
            weight = 1
        if weight is True:
            weight = []
            sc = 1
            for o in range(octaves):
                weight.extend([sc] * intervals)
                sc *= 2

        weight = torch.tensor(weight)
        self.register_buffer('weight', weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        input = self.pyr(input)

        if self.symmetric:
            target = self.pyr(target)
        else:
            target = _fake_pyr(target, self.pyr.octaves, self.pyr.intervals, self.pyr.dim, self.pyr.downsample)

        result = []
        for i in range(len(input)):
            result.append(F.mse_loss(input[i], target[i], reduction=self.reduction))

        if self.reduction == 'none':
            return result

        return (torch.stack(result) * self.weight).sum()
