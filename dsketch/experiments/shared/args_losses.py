import sys

import lpips
import torch
import torch.nn.functional as F

from dsketch.experiments.shared.utils import list_class_names
from dsketch.losses import BlurredMSE, PyramidMSE, DoGPyramidMSE


class _Loss:
    def __init__(self, args):
        if 'variational' in args and args.variational is True:
            self.imagewise = True
        else:
            self.imagewise = False

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, input, target):
        pass


class MSELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, input, target):
        if self.imagewise:
            return F.mse_loss(input, target, reduction='sum') / input.shape[0]
        return F.mse_loss(input, target)


class BCELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, input, target):
        if self.imagewise:
            return F.binary_cross_entropy(input, target, reduction='sum') / input.shape[0]
        return F.binary_cross_entropy(input, target)


class LPIPSLoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

        self.loss = lpips.LPIPS(net=args.net).to(args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--net", help="network for pips [alex or vgg]", type=str, default='vgg', required=False)

    def __call__(self, input, target):
        input2 = (input - 0.5) * 2
        target2 = (target - 0.5) * 2

        if input2.ndim == 3:
            input2 = input2.unsqueeze(0)
            target2 = target2.unsqueeze(0)

        if input2.shape[1] != 3:
            input2 = torch.cat([input2] * 3, dim=1)
            target2 = torch.cat([target2] * 3, dim=1)

        return self.loss(input2, target2)


class PyrMSELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)
        self.loss = PyramidMSE(args.channels, args.octaves, args.intervals, downsample=args.downsample,
                               symmetric=args.no_symmetric).to(args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--octaves", help="number of octaves", type=int, default=3, required=False)
        p.add_argument("--intervals", help="number of intervals", type=int, default=2, required=False)
        p.add_argument("--downsample", help="enable pyramid mode", action='store_true', required=False)
        p.add_argument("--no-symmetric", help="disable symmetric mode", action='store_false', required=False)

    def __call__(self, input, target):
        if input.ndim == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.loss(input, target)


class BlurredMSELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)
        self.loss = BlurredMSE(args.channels, args.blur_sigma, symmetric=args.no_symmetric).to(args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--blur-sigma", help="sigma for blurring in loss", type=float, default=1.0, required=False)
        p.add_argument("--no-symmetric", help="disable symmetric mode", action='store_false', required=False)

    def __call__(self, input, target):
        if input.ndim == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.loss(input, target)


class DoGPyrMSELoss(PyrMSELoss):
    def __init__(self, args):
        args.no_symmetric = None
        super().__init__(args)
        self.loss = DoGPyramidMSE(args.channels, args.octaves, args.intervals, downsample=args.downsample).to(
            args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--octaves", help="number of octaves", type=int, default=3, required=False)
        p.add_argument("--intervals", help="number of intervals", type=int, default=2, required=False)
        p.add_argument("--downsample", help="enable pyramid mode", action='store_true', required=False)


def get_loss(name):
    los = getattr(sys.modules[__name__], name)
    if not issubclass(los, _Loss):
        raise TypeError()
    return los


def loss_choices():
    return list_class_names(_Loss, __name__)


def build_loss(args):
    return get_loss(args.loss)(args)
