import math
import numbers
from typing import Union, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import _int, _size, Number


# noinspection PyTypeChecker
def gaussian_kernel(sigma: Union[_int, _size], dim: _int = 2, kernel_size: Optional[Union[_int, _size]] = None) \
        -> torch.Tensor:
    """Compute a n-dimensional Gaussian kernel.

    The created Gaussian is axis-aligned, but need not be isotropic.

    Args:
        sigma: The standard deviation of the Gaussian along each dimension. Can be a single int for an isotropic
           kernel or tuple with dim elements.
        dim: Number of dimensions.
        kernel_size: The size of the kernel tensor. If None it will be set at the next odd integer above
           floor(8*sigma+1). If it in an int the kernel will have the same size in all dimensions. Can also be a tuple
           with dim elements for a non-square kernel.

    Returns:
        Gaussian kernel tensor
    """
    if kernel_size is None:
        kernel_size = int(8 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    if sigma == 0:
        kernel = torch.zeros([kernel_size] * dim)
        kernel[(int(kernel_size / 2),) * dim] = 1
        return kernel

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    # The kernel is the product of the gaussian of each dimension.
    kernel = 1
    meshgrids: int = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Normalise sum to 1
    kernel = kernel / torch.sum(kernel)

    return kernel


def gaussian_pyramid_kernels(octaves: _int = 3, intervals: _int = 1, init_sigma: Number = 0.5, dim: _int = 2,
                             downsample: bool = False, interval_oversample: _int = 0):
    """Compute gaussian kernels required to incrementally build a pyramid.

    Each kernel increases the scale of the input which it is convolved against from its current scale to the next in the
    pyramid. Pyramids are defined by octaves (number of doublings of sigma from the init_sigma) and intervals which
    defines how many steps are taken within an octave.

    Args:
        octaves: Number of octaves in the pyramid. Each octave represents a doubling of sigma.
        intervals: Number of intervals to break each octave into
        init_sigma: Standard deviation blurring of the input images. Default of 0.5 is the minimum that would be
          possible without aliasing
        dim: number of dimensions for the gaussian
        downsample: if true then each octave is subsampled by taking every other pixel; this also halves the blurring
          required after each octave
        interval_oversample: extra intervals to add beyond the point of doubling

    Returns:
        A list of kernel tensors which incrementally increase the scale of the image to which they are applied. Note
        that these kernels should be applied one after the other to be used correctly: i*k1, i*k1*k2, i*k1*k2*k3, ...
    """
    prev_sigma = init_sigma
    kernels = []

    for j in range(octaves):
        for i in range(intervals + interval_oversample):
            k = 2 ** (1 / intervals)
            sigma = prev_sigma * math.sqrt(k * k - 1)  # this is the amount to increase by
            prev_sigma = prev_sigma * k
            kernels.append(gaussian_kernel(sigma, dim=dim, kernel_size=None))

        if downsample:
            prev_sigma = init_sigma  # downsampling means that effectively each octave starts at init_sigma
        else:
            if interval_oversample == 0:
                assert(abs(prev_sigma - init_sigma * 2 ** (j + 1)) < 1e-7)
            prev_sigma = init_sigma * 2 ** (j + 1)

    return kernels


def _get_conv(dim):
    if dim == 1:
        return F.conv1d
    elif dim == 2:
        return F.conv2d
    elif dim == 3:
        return F.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
        )


class GaussianScaleSpace(nn.Module):
    """Build a Gaussian scale space from a 1d, 2d or 3d tensor.

    Filtering is performed separately for each channel in the input using a depthwise convolution.
    Optionally downsamples each octave to form a pyramid.

    Args:
        channels: Number of channels of the input tensors. Output will have this number of channels as well.
        octaves: Number of octaves in the pyramid. Each octave represents a doubling of sigma.
        intervals: Number of intervals to break each octave into
        init_sigma: Standard deviation blurring of the input images. Default of 0.5 is the minimum that would be
          possible without aliasing
        dim: number of dimensions for the gaussian
        downsample: if true then each octave is subsampled by taking every other pixel
        interval_oversample: extra intervals to add beyond the point of doubling

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
        - Output: List of octaves*intervals tensors. If downsample is False, they will each have the same
            size as the input tensor. If downsample is True then the first intervals tensors will have the same size
            as the input, the next intervals tensors will have half the size in the * dimensions and so forth.
    """
    def __init__(self, channels: _int, octaves: _int = 3, intervals: _int = 1, init_sigma: Number = 0.5, dim: _int = 2,
                 downsample: bool = False, interval_oversample: _int = 0):
        super(GaussianScaleSpace, self).__init__()

        self.octaves = octaves
        self.intervals = intervals
        self.groups = channels
        self.dim = dim
        self.downsample = downsample
        self.interval_oversample = interval_oversample

        flatkernels = gaussian_pyramid_kernels(octaves, intervals, init_sigma, dim, downsample, interval_oversample)

        for i in range(len(flatkernels)):
            kernel = flatkernels[i]
            kernel = kernel.view(1, 1, *kernel.size())
            kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
            self.register_buffer('weight' + str(i), kernel)

        self.conv = _get_conv(dim)

    def forward(self, inp: torch.Tensor) -> List[torch.Tensor]:
        result = [inp]
        c = 0
        for o in range(self.octaves):
            for i in range(self.intervals + self.interval_oversample):
                weight = self.__getattr__('weight' + str(c))
                p = int(weight.shape[-1] / 2)
                padded = F.pad(result[-1], (p, p) * self.dim)
                img = self.conv(padded, weight=weight, groups=self.groups)
                result.append(img)
                c += 1

            if self.downsample:
                idx = -1 - self.interval_oversample
                if self.dim == 1:
                    result.append(result[idx][..., 0::2])
                if self.dim == 2:
                    result.append(result[idx][..., 0::2, 0::2])
                if self.dim == 3:
                    result.append(result[idx][..., 0::2, 0::2, 0::2])

        return result


class GaussianBlur(nn.Module):
    """Apply gaussian smoothing on a 1d, 2d or 3d tensor.

    Filtering is performed separately for each channel in the input using a depthwise convolution.

    Args:
        channels: Number of channels of the input tensors. Output will have this number of channels as well.
        sigma: Standard deviation of gaussian blurring kernel.
        dim: number of dimensions for the gaussian.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
        - Output: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
    """
    def __init__(self, channels: _int, sigma: Number, dim: _int = 2):
        super(GaussianBlur, self).__init__()

        self.groups = channels
        self.dim = dim

        kernel = gaussian_kernel(sigma, dim=dim)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)

        self.conv = _get_conv(dim)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        p = int(self.weight.shape[-1] / 2)
        padded = F.pad(inp, (p, p) * self.dim)
        result = self.conv(padded, weight=self.weight, groups=self.groups)

        return result


class DoGScaleSpace(GaussianScaleSpace):
    """Build a Difference-of-Gaussian scale space from a 1d, 2d or 3d tensor.

    Filtering is performed separately for each channel in the input using a depthwise convolution.
    Optionally downsamples each octave to form a pyramid.

    Args:
        channels: Number of channels of the input tensors. Output will have this number of channels as well.
        octaves: Number of octaves in the pyramid. Each octave represents a doubling of sigma.
        intervals: Number of intervals to break each octave into
        init_sigma: Standard deviation blurring of the input images. Default of 0.5 is the minimum that would be
          possible without aliasing
        dim: number of dimensions for the gaussian
        downsample: if true then each octave is subsampled by taking every other pixel
        interval_oversample: extra intervals to add beyond the point of doubling

    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means dim number of additional dimensions
        - Output: List of octaves*intervals tensors. If downsample is False, they will each have the same
            size as the input tensor. If downsample is True then the first intervals tensors will have the same size
            as the input, the next intervals tensors will have half the size in the * dimensions and so forth.
    """
    def __init__(self, channels: _int, octaves: _int = 3, intervals: _int = 1, init_sigma: Number = 0.5, dim: _int = 2,
                 downsample: bool = False, interval_oversample: _int = 1):
        super(DoGScaleSpace, self).__init__(channels, octaves, intervals, init_sigma, dim, downsample, interval_oversample)

    def forward(self, inp: torch.Tensor) -> List[torch.Tensor]:
        gauss = super(DoGScaleSpace, self).forward(inp)
        result = []

        c = 1
        for o in range(self.octaves):
            for i in range(self.intervals + self.interval_oversample):
                prev = gauss[c-1]
                curr = gauss[c]

                if prev.shape == curr.shape:
                    result.append(prev - curr)

                c += 1

        return result
