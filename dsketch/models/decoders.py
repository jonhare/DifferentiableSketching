from numbers import Number

import torch
import torch.nn as nn

from dsketch.raster.composite import softor
from dsketch.raster.disttrans import line_edt2
from dsketch.raster.raster import exp


class _RasteriserBase(nn.Module):
    def __init__(self, sz):
        super(_RasteriserBase, self).__init__()

        # build the coordinate grid:
        if isinstance(sz, Number.number):
            sz = (sz, sz)

        r = torch.linspace(-1, 1, sz[0])
        c = torch.linspace(-1, 1, sz[1])
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)


class SimplePrimitiveRenderer(_RasteriserBase):
    """
    Simple Primitive decoder for a fixed number of primatives/image and a single scalar rasterisation sigma.

    Args:
        raster: rasterisation function
        comp: composition function
        sz: image size (integer or tuple of two integers)
    """
    def __init__(self, edt2=line_edt2, raster=exp, comp=softor, sz=256):
        super(SimplePrimitiveRenderer, self).__init__(sz)

        self.edt2 = edt2
        self.raster = raster
        self.comp = comp

    def forward(self, params, sigma):
        bs = params.shape[0]
        params = params.view(bs, -1, self.edt2.param_dim, 2)  # -> [batch, n_prims, param_dim, 2]

        rasters = self.raster(self.edt2(params, self.grid), sigma)  # -> [batch, n_prims, sz, sz]

        return self.comp(rasters).unsqueeze(1)  # -> [batch, 1, sz, sz]


class SimpleConfigurablePrimitiveRenderer(_RasteriserBase):
    """
    Simple Primitive decoder for a fixed number of primatives/image and a single scalar rasterisation sigma.

    Args:
        raster: rasterisation function
        comp: composition function
        sz: image size (integer or tuple of two integers)
    """
    def __init__(self, edt2=line_edt2, raster=exp, comp=softor, sz=256):
        super(SimpleConfigurablePrimitiveRenderer, self).__init__(sz)

        self.edt2 = edt2
        self.raster = raster
        self.comp = comp

    def forward(self, shape_params, colour_params, thickness_params, sigma):
        bs = shape_params.shape[0]
        params = shape_params.view(bs, -1, self.edt2.param_dim, 2)  # -> [batch, n_prims, param_dim, 2]

        rasters = self.raster(self.edt2(params, self.grid), sigma)  # -> [batch, n_prims, sz, sz]

        return self.comp(rasters).unsqueeze(1)  # -> [batch, 1, sz, sz]
