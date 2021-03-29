import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.types import _int, Number

from ..utils import GaussianBlur


class BlurredMSE(_Loss):
    """Blurred Mean Squared Error loss.

    This loss function is basically mean-squared error, but with a built-in gaussian receptive field to allow
    surrounding pixels to contribute to the loss of an individual pixel. This is achieved by blurring the input and
    optionally the target with a Gaussian blur (applied independently to each channel) before computing the MSE loss.

    This could for example be useful in the case where a reconstruction error of images is required, but the input is
    constrained to be near black and white whilst the target is greyscale; with the blurred loss the perceptual effect
    of patterns of black and white pixels creating percieved grey levels would be captured.

    Args:
        channels: Number of channels of the input tensors. Output will have this number of channels as well.
        sigma: Standard deviation of gaussian blurring kernel.
        dim: number of dimensions for the gaussian.
        symmetric: if False, only the input is blurred. If True, then the target is also blurred.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
        - Target: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
    """
    def __init__(self, channels: _int, sigma: Number, dim: _int = 2, symmetric: bool = False,
                 reduction: str = 'mean'):
        super(BlurredMSE, self).__init__(None, None, reduction)

        self.blur = GaussianBlur(channels, sigma, dim)
        self.symmetric = symmetric

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.blur(input)

        if self.symmetric:
            target = self.blur(target)

        return F.mse_loss(input, target, reduction=self.reduction)
