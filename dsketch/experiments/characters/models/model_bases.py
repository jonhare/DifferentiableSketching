import importlib
from abc import ABC, abstractmethod

import torch.nn as nn
import torchbearer
from torchbearer.callbacks import decorators as callbacks

from dsketch.experiments.shared import metrics
from dsketch.raster.composite import softor
from dsketch.raster.raster import exp, nearest_neighbour, compute_nearest_neighbour_sigma_bres

MU = torchbearer.state_key('mu')
LOGVAR = torchbearer.state_key('logvar')


class _Base(nn.Module, ABC):
    @classmethod
    def add_args(cls, p):
        cls._add_args(p)

    @staticmethod
    @abstractmethod
    def _add_args(p):
        pass

    @staticmethod
    @abstractmethod
    def create(args):
        pass

    @abstractmethod
    def forward(self, x, state=None):
        pass

    def get_callbacks(self, args):
        """return any additional torchbearer callbacks to add during training"""
        return []


class Encoder(_Base, ABC):
    @classmethod
    def add_args(cls, p):
        cls._add_args(p)
        p.add_argument("--latent-size", help="size of latent space", type=int, default=64, required=False)


class Decoder(_Base, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @abstractmethod
    def decode_to_params(self, inp):
        pass

    @abstractmethod
    def create_edt2(self, params):
        pass

    def raster_soft(self, edt2, sigma2):
        rasters = exp(edt2, sigma2)
        return softor(rasters, keepdim=True)

    def raster_hard(self, edt2):
        rasters = nearest_neighbour(edt2.detach(), compute_nearest_neighbour_sigma_bres(self.grid))
        return softor(rasters, keepdim=True)

    @abstractmethod
    def get_sigma2(self, params):
        pass

    @classmethod
    def add_args(cls, p):
        cls._add_args(p)
        p.add_argument("--contrast", help="scale result constrast", type=float, default=1, required=False)

    def forward(self, inp, state=None):
        params = self.decode_to_params(inp)
        sigma2 = self.get_sigma2(params)
        edt2 = self.create_edt2(params)
        images = self.raster_soft(edt2, sigma2)

        if state is not None:
            state[metrics.HARDRASTER] = self.raster_hard(edt2)
            state[metrics.SQ_DISTANCE_TRANSFORM] = edt2

        return images * self.args.contrast


class AdjustableSigmaMixin:
    """
    Provides a sigma2 parameter that isn't learned and can be adjusted externally
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_sigma2(self, params=None):
        return self.sigma2

    def set_sigma2(self, value):
        self.sigma2 = value

    @staticmethod
    def _add_args(p):
        p.add_argument("--sigma2", "--init-sigma2", help="sigma^2 value for drawing lines", type=float, default=1e-2,
                       required=False)
        p.add_argument("--final-sigma2", help="final sigma^2 value for drawing lines; only used is sigma2_step is >0",
                       type=float, default=1e-2, required=False)
        p.add_argument("--sigma2-factor", type=float, required=False,
                       help="factor to multiply sigma^2 by every sigma2-step epochs.", default=0.5)
        p.add_argument("--sigma2-step", type=int, required=False,
                       help="number of epochs between changes in sigma^2", default=-1)

    def get_callbacks(self, args):
        self.set_sigma2(args.sigma2)

        @callbacks.on_end_epoch
        def change_sigma(state):
            if args.sigma2_step > 0 and state[torchbearer.EPOCH] % args.sigma2_step == 0:
                current = self.get_sigma2()
                if current > args.final_sigma2:
                    self.set_sigma2(max(current * args.sigma2_factor, args.final_sigma2))

        return [change_sigma]


def get_model(name):
    # load a model class by name
    module = importlib.import_module(__package__)
    return getattr(module, name)
