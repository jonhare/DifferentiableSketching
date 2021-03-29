"""
Functions for computing squared (Euclidean) distance transforms of points, line segments and curves.
"""

import functools
from typing import Callable, Union, Optional

import torch
from torch.types import Number, _int


def attrs(**atts):
    def with_attrs(f):
        for k, v in atts.items():
            setattr(f, k, v)
        return f

    return with_attrs


@attrs(param_dim=1)
def point_edt2(point: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Batched version of a Squared Euclidean Distance Transform for a D-dimensional point.

    Args:
      point: torch.Tensor of size [batch, *, D]. Each element is interpreted as a D-dimensional point (e.g. [row, col]
        or [depth, row, col]) and * represents any number of additional dimensions (eg. channels or images or both).
      grid: torch.Tensor of size [*D, D], where *D represents D elements defining a lattice that defines the coordinates
        of each output pixel/voxel.

    Returns:
      a torch.Tensor of size [batch, *, *D] representing the EDT^2 of each point in the input batch where *
      represents any additional dimensions from the input, and *D is the size of each dimension of the lattice.

    """
    inshape = point.shape
    outshape = (*inshape[0:-1], *grid.shape[0:-1])
    dim = len(grid.shape) - 1
    point = point.view(-1, *[1] * dim, dim)

    # need to replicate the grid for each item in the batch
    grid = grid.expand(point.shape[0], *grid.shape)

    pl = (grid - point)
    d = (pl * pl).sum(dim=-1)

    d = d.view(outshape)

    return d


@attrs(param_dim=2)
def line_edt2(line: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Batched version of a Squared Euclidean Distance Transform for a D-dimensional line segment.

    Args:
      line: torch.Tensor of size [batch, *, 2, D]. Each element is interpreted as a D dimensional coordinate; the first
        (in [..., 0, D] is the start of the line, and the second (in [..., 1, D]) is the end. * represents any
        number of additional dimensions (eg. channels or images or both).
      grid: torch.Tensor of size [*D, D], where *D represents D elements defining a lattice that defines the coordinates
        of each output pixel/voxel.

    Returns:
      a torch.Tensor of size [batch, *, *D] representing the EDT^2 of each point in the input batch where *
      represents any additional dimensions from the input, and *D is the size of each dimension of the lattice.

    """

    inshape = line.shape
    outshape = (*inshape[0:-2], *grid.shape[0:-1])
    dim = len(grid.shape) - 1

    # Based off ideas from https://monkeyproofsolutions.nl/wordpress
    # /how-to-calculate-the-shortest-distance-between-a-point-and-a-line/
    #
    # Key idea is that we compute 3 maps:
    # 1. distance from the start point (this will be isotropic)
    # 2. distance from the end point (this will be isotropic)
    # 3. distance perpendicular to the line (not the line segment; this is a ridge)
    # The three maps are then combined so that perpendicular to the segment we use
    # the ridge map, and beyond perpendicular at the ends of the lines we use the
    # relevant (closest) isotropic map
    a = line[..., 0, :]  # start
    b = line[..., 1, :]  # end

    a = a.view(-1, *[1] * dim, dim)
    b = b.view(-1, *[1] * dim, dim)
    m = b - a  # vector from a to b

    # need to replicate the grid for each item in the (expanded) batch
    grid = grid.expand(a.shape[0], *grid.shape)

    pa = (grid - a)
    pb = (grid - b)
    t0 = (pa * m).sum(dim=-1) / ((m * m).sum(dim=-1) + torch.finfo().eps)  # t0 is <=0 before the start point,
    # >0 and <1 between start and end,
    # >=1 after the end point

    patm = grid - (a + t0.unsqueeze(-1).expand(*t0.shape, dim) * m)

    # not sure if there is a more efficient way of doing the following? it's
    # just masking the correct bits of the three maps and adding everything together
    d = (pa * pa).sum(dim=-1) * (t0 <= 0) + (pb * pb).sum(dim=-1) * (t0 >= 1) + (patm * patm).sum(-1) * (t0 > 0) * (
            t0 < 1)

    d = d.view(outshape)

    return d


@attrs(param_dim=2)
def linear_bezier(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a batch of linear bezier curves at different points.
    
    This is predominantly for testing, as it would be much more efficient to utilise line segments in a distance
    transform computation.

    Args:
      params: torch.Tensor of size [batch, 2, D]. Each batch item is interpreted as two points corresponding to
        the start point and the end point. D is the number of dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the
        curve at.

    Returns:
      a torch.Tensor of size [batch, nvalues, D] representing the points along each curve in the batch at the
      respective values of t.

    """
    p0 = params[:, 0].unsqueeze(1)
    p1 = params[:, 1].unsqueeze(1)
    return p0 + t * (p1 - p0)


@attrs(param_dim=3)
def quadratic_bezier(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a batch of quadratic bezier curves at different points.

    Args:
      params: torch.Tensor of size [batch, 3, D]. Each batch item is interpreted as four points corresponding to
        the start point, the first control point, the second control point and the end point. D is the number of
        dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the
        curve at.

    Returns:
      a torch.Tensor of size [batch, nvalues, D] representing the points along each curve in the batch at the
      respective values of t.

    """
    p0 = params[:, 0].unsqueeze(1)
    p1 = params[:, 1].unsqueeze(1)
    p2 = params[:, 2].unsqueeze(1)

    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


@attrs(param_dim=4)
def cubic_bezier(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a batch of cubic bezier curves at different points.

    Args:
      params: torch.Tensor of size [batch, 4, D]. Each batch item is interpreted as four points corresponding to
        the start point, the first control point, the second control point and the end point. D is the number of
        dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the
         curve at.

    Returns:
      a torch.Tensor of size [batch, nvalues, D] representing the points along each curve in the batch at the
      respective values of t.

    """
    p0 = params[:, 0].unsqueeze(1)
    p1 = params[:, 1].unsqueeze(1)
    p2 = params[:, 2].unsqueeze(1)
    p3 = params[:, 3].unsqueeze(1)

    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


@attrs(param_dim=4)
def centripetal_catmull_rom_spline(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a batch of Centripetal Catmull-Rom Splines at different points.

    Args:
      params: torch.Tensor of size [batch, 4, D]. Each batch item is interpreted as four points corresponding to
              the points along the spline. The curve itself is bounded by the second and third points; the first and
              last only define the directions. D is the number of dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the
         curve at.

    Returns:
      a torch.Tensor of size [batch, nvalues, D] representing the points along each curve in the batch at the
      respective values of t.

    """
    return catmull_rom_spline(params, t, alpha=0.5)


@attrs(param_dim=4)
def uniform_catmull_rom_spline(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a batch of Uniform Catmull-Rom Splines at different points.

    Args:
      params: torch.Tensor of size [batch, 4, D]. Each batch item is interpreted as four points corresponding to
              the points along the spline. The curve itself is bounded by the second and third points; the first and
              last only define the directions. D is the number of dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the
         curve at.

    Returns:
      a torch.Tensor of size [batch, nvalues, D] representing the points along each curve in the batch at the
      respective values of t.

    """
    return catmull_rom_spline(params, t, alpha=0.0)


@attrs(param_dim=4)
def chordal_catmull_rom_spline(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a batch of Chordal Catmull-Rom Splines at different points.

    Args:
      params: torch.Tensor of size [batch, 4, D]. Each batch item is interpreted as four points corresponding to
              the points along the spline. The curve itself is bounded by the second and third points; the first and
              last only define the directions. D is the number of dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the
         curve at.

    Returns:
      a torch.Tensor of size [batch, nvalues, D] representing the points along each curve in the batch at the
      respective values of t.

    """
    return catmull_rom_spline(params, t, alpha=1.0)


@attrs(param_dim=4)
def catmull_rom_spline(params: torch.Tensor, t: torch.Tensor, alpha: Number = 0.5) -> torch.Tensor:
    """Evaluate a batch of Catmull-Rom Splines at different points.

    Args:
      params: torch.Tensor of size [batch, 4, D]. Each batch item is interpreted as four points corresponding to
              the points along the spline. The curve itself is bounded by the second and third points; the first and
              last only define the directions. D is the number of dimensions.
      t: torch.Tensor of size [batch, nvalues, 1] or [nvalues, 1] that defines the values of t to evaluate the curve at.
      alpha: number between 0 and 1 defining the type of spline knot parameterisation. A value of 0 is the standard
             uniform Catmull-Rom spline; 0.5 is centripetal; and, 1.0 is the chordal spline. Defaults to the centripetal
             spline.

    Returns:
      a torch.Tensor of size [batch, nvalues, 2] representing the points along each curve in the batch at the
      respective values of t.

    """

    if t.ndim == 2:
        t = t.expand(params.shape[0], *t.shape)

    p0 = params[:, 0].unsqueeze(1)
    p1 = params[:, 1].unsqueeze(1)
    p2 = params[:, 2].unsqueeze(1)
    p3 = params[:, 3].unsqueeze(1)

    # Premultiplied power constant for the following tj() function.
    alpha = alpha / 2

    def tj(ti, pi, pj):
        return ((pi - pj) ** 2).sum(dim=-1) ** alpha + ti

    t0 = torch.zeros(params.shape[0], 1, device=params.device)
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    t0 = t0.unsqueeze(1)
    t1 = t1.unsqueeze(1)
    t2 = t2.unsqueeze(1)
    t3 = t3.unsqueeze(1)

    t = t1 + t * (t2 - t1)  # t is mapped to be between t1 and t2

    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3

    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def curve_edt2_bruteforce(params: torch.Tensor, grid: torch.Tensor, iters: _int, slices: _int,
                          cfcn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cubic_bezier,
                          mint: Union[Number, torch.Tensor] = 0, maxt: Union[Number, torch.Tensor] = 1,
                          mindists: Optional[torch.Tensor] = None, _outshape: Optional[torch.Tensor] = None):
    """Batched version of a Squared Euclidean Distance Transform for a parameterised curve, c(t) for 0≤t≤1.

    This version implements a recursive brute force approach in which the distance is computed at `slices` points
    (mint-maxt)/slices apart along the curve. The value of t which produces the minimum distance is selected for the
    next iteration whereby the search is repeated, but localised at that point +/-((mint-maxt)/slices). This process
    continues until the max number of iterations is reached.

    Args:
      params: torch.Tensor of size [batch, *, P, D]. Each batch item represents the parameters describing a parametric
              curve with P control points in D dimensions.
      grid: torch.Tensor of size [*D, D], where *D represents D elements defining a lattice that defines the coordinates
        of each output pixel/voxel.
      iters: integer number of iterations to perform.
      slices: integer number of slices of the curve per iteration.
      cfcn: function mapping the curve parameters and sampled values of t to coordinates on the curve (Default value =
            cubic_bezier)
      mint: number or torch.Tensor of size [batch, rows, cols] describing the minimum value of t to start the
            search for for each point in grid (Default value = 0)
      maxt: number or torch.Tensor of size [batch, rows, cols] describing the maximum value of t to start the
            search for for each point in grid (Default value = 1)
      mindists: torch.Tensor of size [batch, rows, cols] storing intermediate best-guesses of the distance for each
                point in grid. Can be None, in which case it will be created automatically. (Default value = None)
      _outshape: [internal - ignored on the first recursive call] desired output tensor shape

    Returns:
      a torch.Tensor of size [batch, rows, cols] representing the EDT^2 of each curve in the input batch

    """
    # if we're on the first call:
    if grid.ndim == grid.shape[-1] + 1:
        inshape = params.shape
        _outshape = (*inshape[0:-2], *grid.shape[0:-1])
        # need to replicate the grid for each item in the batch
        grid = grid.expand(params.shape[0], *grid.shape)

    if mindists is None:
        mindists = torch.ones(*grid.shape[0:-1], device=grid.device) * torch.finfo().max
    if iters == 0:
        return mindists.view(_outshape)

    if not isinstance(mint, torch.Tensor):
        mint = torch.ones(*grid.shape[0:-1], device=grid.device) * mint
    if not isinstance(maxt, torch.Tensor):
        maxt = torch.ones(*grid.shape[0:-1], device=grid.device) * maxt

    tick = (maxt - mint) / slices

    t = mint.clone()

    for i in range(slices):
        curve = cfcn(params, t.view(grid.shape[0], -1, 1))
        curve = curve.view(*grid.shape[0:-1], grid.ndim - 2)

        diff = curve - grid
        dist = (diff * diff).sum(dim=-1)
        mask = dist < mindists

        # update minimum distances and minimum t's
        mint = mask * t + ~mask * mint
        mindists = mask * dist + ~mask * mindists

        t = t + tick

    maxt = mint + tick
    mint = mint - tick

    maxt = (maxt < 1) * maxt + (maxt >= 1)
    mint = (mint >= 0) * mint

    return curve_edt2_bruteforce(params, grid, iters - 1, slices, cfcn, mint, maxt, mindists, _outshape)


def create_curve_edt2_bruteforce(iters: _int, slices: _int,
                                 cfcn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cubic_bezier,
                                 mint: Union[Number, torch.Tensor] = 0, maxt: Union[Number, torch.Tensor] = 1) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    edt2 = functools.partial(curve_edt2_bruteforce, iters=iters, slices=slices, cfcn=cfcn, mint=mint, maxt=maxt)
    edt2.param_dim = cfcn.param_dim

    return edt2


def curve_edt2_polyline(params: torch.Tensor, grid: torch.Tensor, segments: _int,
                        cfcn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cubic_bezier):
    """Batched version of a Squared Euclidean Distance Transform for a parameterised curve, c(t) for 0≤t≤1.

    This version implements a simple polyline approximation to the curve by breaking it into uniform segments (note that this
    is not necessarily a sensible thing to do, for example when there is high curvature)

    Args:
      params: torch.Tensor of size [batch, *, P, D]. Each batch item represents the parameters describing a parametric
              curve with P control points in D dimensions.
      grid: torch.Tensor of size [*D, D], where *D represents D elements defining a lattice that defines the coordinates
        of each output pixel/voxel.
      segments: integer number of polyline segments to use in the approximation.
      cfcn: function mapping the curve parameters and sampled values of t to coordinates on the curve (Default value =
            cubic_bezier)

    Returns:
      a torch.Tensor of size [batch, rows, cols] representing the EDT^2 of each curve in the input batch

    """

    # flatten params to batch^, P, D
    inshape = params.shape
    _outshape = (*inshape[0:-2], *grid.shape[0:-1])
    params = params.view(-1, params.shape[-2], params.shape[-1])

    # values of t to sample
    t = (torch.arange(segments + 1) / segments).unsqueeze(1).to(params.device)
    pts = cfcn(params, t)  # batch^, segments+1, D

    # arrange into connected segments
    coordpairs = torch.stack([torch.arange(0, segments, 1), torch.arange(1, segments + 1, 1)], dim=1)
    lines = torch.stack((pts[:, coordpairs[:, 0]], pts[:, coordpairs[:, 1]]), dim=-2)  # [batch^, segments, 2, D]

    # Compute line_edt2
    edt2s = line_edt2(lines, grid)

    # take element-wise min across the segments dim of the resultant edt2s
    edt2s, _ = edt2s.min(dim=1)

    # reshape to put back the batch into addtional dimes
    edt2s = edt2s.view(_outshape)

    # return
    return edt2s
