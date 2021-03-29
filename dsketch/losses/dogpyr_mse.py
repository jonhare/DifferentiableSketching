from typing import Union, Sequence, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.types import _int, Number

from ..utils import DoGScaleSpace


class DoGPyramidMSE(_Loss):
    """Mean Squared Error loss over a Difference of Gaussian scale space or pyramid.

    This loss function is basically mean-squared error applied to scale-space difference-of-gaussian representations of
    the target and input images. The representation can by a pure scale space (no downsampling) or pyramidal
    (downsampling very octave).

    Intuitively this loss should favor reconstructions that capture the broad structure of the target (strong edges for
    example).

    Args:
        channels: Number of channels of the input tensors. Output will have this number of channels as well.
        octaves: Number of octaves in the pyramid. Each octave represents a doubling of sigma.
        intervals: Number of intervals to break each octave into
        init_sigma: Standard deviation blurring of the input images. Default of 0.5 is the minimum that would be
            possible without aliasing
        dim: number of dimensions for the (difference of) gaussian
        downsample: if true then each octave is subsampled by taking every other pixel
        weight: if False, then each scale is weighted according to its size when computing the mean; if true then each
            time downsampling is used, the means of those downsampled layers is multiplied by 4^s before summation so
            that the number of pixels is equal in each scale. Can also be a tensor or sequence of length
            octaves*intervals+1 to specify a particular weight for each corresponding scale. Has no effect if reduction
            is 'none'.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
        - Target: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
    """

    def __init__(self, channels: _int, octaves: _int = 3, intervals: _int = 1, init_sigma: Number = 0.5, dim: _int = 2,
                 downsample=False, weight: Union[bool, Sequence[Number], torch.Tensor] = False,
                 reduction: str = 'mean'):
        super(DoGPyramidMSE, self).__init__(None, None, reduction)

        self.pyr = DoGScaleSpace(channels, octaves, intervals, init_sigma, dim, downsample)

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
        target = self.pyr(target)

        result = []
        for i in range(len(input)):
            result.append(F.mse_loss(input[i], target[i], reduction=self.reduction))

        if self.reduction == 'none':
            return result

        return (torch.stack(result) * self.weight).sum()
