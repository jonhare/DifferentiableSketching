import torch
import torch.nn as nn

from dsketch.experiments.characters.models.model_bases import AdjustableSigmaMixin, Decoder
from dsketch.experiments.shared import metrics
from dsketch.raster.composite import softor
from dsketch.raster.disttrans import catmull_rom_spline, curve_edt2_bruteforce, curve_edt2_polyline, line_edt2, point_edt2
from dsketch.raster.raster import exp


class SinglePassCRSDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, args, npoints=4, nlines=1, input=64, hidden=64, hidden2=256, sz=28, edt_approx='polyline',
                 sigma2=1e-2):
        super().__init__(args)

        # build the coordinate grid:
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

        # this is a list of quads of "connections" 0-1-2-3, 1-2-3-4, 2-3-4-5, ...
        self.coordpairs = torch.stack([torch.arange(0, npoints - 3, 1),
                                       torch.arange(1, npoints - 2, 1),
                                       torch.arange(2, npoints - 1, 1),
                                       torch.arange(3, npoints, 1)], dim=1)
        self.npoints = npoints
        self.latent_to_points = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, nlines * npoints * 2),
            nn.Tanh()
        )

        self.edt_approx = edt_approx
        self.sigma2 = sigma2

    @staticmethod
    def _add_args(p):
        p.add_argument("--npoints", help="number of control points", type=int, default=16, required=False)
        p.add_argument("--nlines", help="number of independent lines", type=int, default=1, required=False)
        p.add_argument("--decoder-hidden", help="decoder hidden size", type=int, default=64, required=False)
        p.add_argument("--decoder-hidden2", help="decoder hidden2 size", type=int, default=256, required=False)
        p.add_argument("--edt-approximation", help="approximation to use for computing edt", type=str,
                       default="polyline", required=False, choices=['polyline', 'bruteforce'])
        AdjustableSigmaMixin._add_args(p)

    @staticmethod
    def create(args):
        return SinglePassCRSDecoder(args, npoints=args.npoints, nlines=args.nlines, input=args.latent_size,
                                    hidden=args.decoder_hidden, hidden2=args.decoder_hidden2, sz=args.size,
                                    edt_approx=args.edt_approximation, sigma2=args.sigma2)

    def decode_to_params(self, inp):
        # the latent_to_points process will map the input latent vector to control points for the CRS
        bs = inp.shape[0]
        pts = self.latent_to_points(inp)  # [batch, npoints*2]
        pts = pts.view(bs, -1, self.npoints, 2)  # expand -> [batch, nlines, npoints, 2]

        # compute all valid permutations of line start and end points
        lines = torch.cat((pts[:, :, self.coordpairs[:, 0]],
                           pts[:, :, self.coordpairs[:, 1]],
                           pts[:, :, self.coordpairs[:, 2]],
                           pts[:, :, self.coordpairs[:, 3]]), dim=-1)  # [batch, nlines, 8]

        lines = lines.view(bs, -1, 4, 2)  # flatten -> [batch, nlines, 4, 2]
        return lines

    def create_edt2(self, lines):
        if self.edt_approx == 'polyline':
            edt2 = curve_edt2_polyline(lines, self.grid, 10, cfcn=catmull_rom_spline)
        elif self.edt_approx == 'bruteforce':
            edt2 = curve_edt2_bruteforce(lines, self.grid, 2, 10, cfcn=catmull_rom_spline)
        else:
            raise
        return edt2


class SinglePassPolyLineDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, args, npoints=8, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2):
        super().__init__(args)

        # build the coordinate grid:
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

        # this is a list of pairs of "connections" 0-1, 1-2, 2-3, ...
        self.coordpairs = torch.stack([torch.arange(0, npoints - 1, 1), torch.arange(1, npoints, 1)], dim=1)

        self.latent_to_linecoord = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, npoints * 2),
            nn.Tanh()
        )

        self.sigma2 = sigma2

    @staticmethod
    def _add_args(p):
        p.add_argument("--npoints", help="number of control points", type=int, default=16, required=False)
        p.add_argument("--decoder-hidden", help="decoder hidden size", type=int, default=64, required=False)
        p.add_argument("--decoder-hidden2", help="decoder hidden2 size", type=int, default=256, required=False)
        AdjustableSigmaMixin._add_args(p)

    @staticmethod
    def create(args):
        return SinglePassPolyLineDecoder(args, npoints=args.npoints, input=args.latent_size, hidden=args.decoder_hidden,
                                         hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2)

    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]
        pts = self.latent_to_linecoord(inp)  # [batch, npoints*2]
        pts = pts.view(bs, -1, 2)  # expand -> [batch, npoints, 2]

        lines = torch.stack((pts[:, self.coordpairs[:, 0]], pts[:, self.coordpairs[:, 1]]), dim=-2)
        # -> [batch, nlines, 2, 2]

        return lines

    def create_edt2(self, lines):
        edt2 = line_edt2(lines, self.grid)

        return edt2


class SinglePassPolyConnectDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, args, npoints=8, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2, allow_points=True,
                 hard=False):
        super().__init__(args)

        self.hard = hard

        # build the coordinate grid:
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

        # if we allow points, we compute the upper-triangular part of the symmetric connection
        # matrix including the diagonal. If points are not allowed, we don't need the diagonal values
        # as they would be implictly zero
        if allow_points:
            nlines = int((npoints ** 2 + npoints) / 2)
        else:
            nlines = int(npoints * (npoints - 1) / 2)
        self.coordpairs = torch.combinations(torch.arange(0, npoints, dtype=torch.long), r=2,
                                             with_replacement=allow_points)

        # shared part of the encoder
        self.enc1 = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU())
        # second part for computing npoints 2d coordinates (using tanh because we use a -1..1 grid)
        self.enc_pts = nn.Sequential(
            nn.Linear(hidden2, npoints * 2),
            nn.Tanh())
        # second part for computing upper triangular part of the connection matrix
        self.enc_con = nn.Sequential(
            nn.Linear(hidden2, nlines),
            nn.Sigmoid())

        self.sigma2 = sigma2

    @staticmethod
    def _add_args(p):
        SinglePassPolyLineDecoder._add_args(p)
        p.add_argument("--allowPoints",
                       help="If we allow points, we compute the upper-triangular part of the symmetric connection",
                       type=bool, default=True, required=False)
        p.add_argument("--hard", help="force lines to be black or white rather than grey using ST-threshold",
                       default=False, action='store_true')

    @staticmethod
    def create(args):
        return SinglePassPolyConnectDecoder(args, npoints=args.npoints, input=args.latent_size,
                                            hidden=args.decoder_hidden,
                                            hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2,
                                            allow_points=args.allowPoints, hard=args.hard)

    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]
        z = self.enc1(inp)
        pts = self.enc_pts(z)  # [batch, npoints*2]
        pts = pts.view(bs, -1, 2)  # expand -> [batch, npoints, 2]

        conn = self.enc_con(z)

        # compute all valid permuations of line start and end points
        lines = torch.stack((pts[:, self.coordpairs[:, 0]], pts[:, self.coordpairs[:, 1]]),
                            dim=-2)  # [batch, nlines, 2, 2]

        return lines, conn

    def create_edt2(self, lines):
        edt2 = line_edt2(lines, self.grid)

        return edt2

    def raster_soft(self, edt2, conn):
        rasters = exp(edt2, self.sigma2)

        connect = conn.view(edt2.shape[0], -1, 1, 1)

        if self.hard:
            connect = connect + ((connect > 0.5).float() - connect).detach()

        rasters = rasters * connect
        return softor(rasters, keepdim=True)

    def forward(self, inp, state=None):
        params, conn = self.decode_to_params(inp)
        edt2 = self.create_edt2(params)
        images = self.raster_soft(edt2, conn)

        if state is not None:
            state[metrics.HARDRASTER] = self.raster_hard(edt2)
            state[metrics.SQ_DISTANCE_TRANSFORM] = edt2

        return images * self.args.contrast


class SinglePassSimpleLineDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, args, nlines=5, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2):
        super().__init__(args)

        # build the coordinate grid:
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

        self.latent_to_linecoord = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, nlines * 4),
            nn.Tanh()
        )

        self.sigma2 = sigma2

    @staticmethod
    def _add_args(p):
        p.add_argument("--nlines", help="number of lines", type=int, default=5, required=False)
        p.add_argument("--decoder-hidden", help="decoder hidden size", type=int, default=64, required=False)
        p.add_argument("--decoder-hidden2", help="decoder hidden2 size", type=int, default=256, required=False)
        AdjustableSigmaMixin._add_args(p)

    @staticmethod
    def create(args):
        return SinglePassSimpleLineDecoder(args, nlines=args.nlines, input=args.latent_size, hidden=args.decoder_hidden,
                                           hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2)

    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]

        lines = self.latent_to_linecoord(inp)  # [batch, nlines*4]
        lines = lines.view(bs, -1, 2, 2)  # expand -> [batch, nlines, 2, 2]

        return lines

    def create_edt2(self, lines):
        edt2 = line_edt2(lines, self.grid)

        return edt2


class SinglePassBezierConnectDecoder(SinglePassPolyConnectDecoder):
    def __init__(self, args, npoints=8, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2, allow_points=True,
                 hard=False):
        super().__init__(args, npoints, input, hidden, hidden2, sz, sigma2, allow_points, hard)

        # second part for computing npoints 2d coordinates (using tanh because we use a -1..1 grid),
        # and the corresponding control point
        self.enc_pts = nn.Sequential(
            nn.Linear(hidden2, npoints * 4),
            nn.Tanh())

    @staticmethod
    def create(args):
        return SinglePassBezierConnectDecoder(args, npoints=args.npoints, input=args.latent_size,
                                              hidden=args.decoder_hidden, hidden2=args.decoder_hidden2, sz=args.size,
                                              sigma2=args.sigma2, allow_points=args.allowPoints, hard=args.hard)

    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]
        z = self.enc1(inp)
        pts = self.enc_pts(z)  # [batch, npoints*2]
        pts = pts.view(bs, -1, 2, 2)  # expand -> [batch, npoints, 2, 2]

        conn = self.enc_con(z)

        # compute all valid permuations of line start and end points
        lines = torch.stack(
            (pts[:, self.coordpairs[:, 0]][:, :, 0],
             pts[:, self.coordpairs[:, 0]][:, :, 1],
             2 * pts[:, self.coordpairs[:, 1]][:, :, 0] - pts[:, self.coordpairs[:, 1]][:, :, 1],  # reflected
             pts[:, self.coordpairs[:, 1]][:, :, 0]), dim=-2)  # [batch, nlines, 4, 2]

        return lines, conn

    def create_edt2(self, lines):
        edt2 = curve_edt2_polyline(lines, self.grid, 10)

        return edt2


class SinglePassSimpleBezierDecoder(SinglePassSimpleLineDecoder):
    def __init__(self, args, nlines=5, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2, segments=1, smooth=True):
        super().__init__(args, nlines, input, hidden, hidden2, sz, sigma2)

        self.segments = segments
        self.smooth = smooth
        if smooth:
            npoints = (4 + (segments - 1) * 2)  # 4 points + 2 for each intermediate [pt, cp]

            # this is a list of pairs of "connections" 0-1-2-3, 3-2-4-5, 5-4-6-7, ...
            self.coordpairs = torch.stack([torch.arange(0, npoints - 3, 2),
                                           torch.arange(1, npoints - 2, 2),
                                           torch.arange(2, npoints - 1, 2),
                                           torch.arange(3, npoints - 0, 2)], dim=1)
            self.coordpairs[1:, 0] += 1
            self.coordpairs[1:, 1] -= 1
        else:
            npoints = (4 + (segments - 1) * 3)  # 4 points + 3 for each intermediate [pt, cp1, cp2]

            # this is a list of pairs of "connections" 0-1-2-3, 3-4-5-6, 6-7-8-9, ...
            self.coordpairs = torch.stack([torch.arange(0, npoints - 3, 3),
                                           torch.arange(1, npoints - 2, 3),
                                           torch.arange(2, npoints - 1, 3),
                                           torch.arange(3, npoints - 0, 3)], dim=1)

        self.npoints = npoints

        self.latent_to_linecoord = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2 * npoints * nlines),  # pts are 2d
            nn.Tanh()
        )

    @staticmethod
    def _add_args(p):
        SinglePassSimpleLineDecoder._add_args(p)
        p.add_argument("--smooth", help="number of lines", default=False, required=False, action='store_true')
        p.add_argument("--segments", help="number of curves to join", type=int, default=1, required=False)

    @staticmethod
    def create(args):
        return SinglePassSimpleBezierDecoder(args, nlines=args.nlines, input=args.latent_size,
                                             hidden=args.decoder_hidden, hidden2=args.decoder_hidden2, sz=args.size,
                                             sigma2=args.sigma2, segments=args.segments, smooth=args.smooth)

    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]

        lines = self.latent_to_linecoord(inp)  # [batch, nlines*4]

        lines = lines.view(bs, -1, self.npoints, 2)  # expand -> [batch, npoints, 2]
        lines = torch.stack((lines[:, :, self.coordpairs[:, 0]], lines[:, :, self.coordpairs[:, 1]],
                             lines[:, :, self.coordpairs[:, 2]], lines[:, :, self.coordpairs[:, 3]]), dim=-2)
        # -> [batch, nlines, nsegments, 4, 2]

        if self.smooth:
            # reflect control point vectors
            lines[:, :, 1:, 1] = 2 * lines[:, :, 1:, 0] - lines[:, :, 1:, 1]

        lines = lines.view(bs, -1, 4, 2)  # flatten out the segments and lines

        return lines

    def create_edt2(self, lines):
        edt2 = curve_edt2_polyline(lines, self.grid, 10)

        return edt2
    
class SinglePassPointsDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, args, npoints=50, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2):
        super().__init__(args)

        # build the coordinate grid:
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

        self.latent_to_pointscoord = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, npoints * 2),
            nn.Tanh()
        )

        self.sigma2 = sigma2

    @staticmethod
    def _add_args(p):
        p.add_argument("--npoints", help="number of points", type=int, default=50, required=False)
        p.add_argument("--decoder-hidden", help="decoder hidden size", type=int, default=64, required=False)
        p.add_argument("--decoder-hidden2", help="decoder hidden2 size", type=int, default=256, required=False)
        AdjustableSigmaMixin._add_args(p)

    @staticmethod
    def create(args):
        return SinglePassPointsDecoder(args, npoints=args.npoints, input=args.latent_size, hidden=args.decoder_hidden,
                                           hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2)

    def decode_to_params(self, inp):
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = inp.shape[0]

        points = self.latent_to_pointscoord(inp)  # [batch, nlines*4]
        points = points.view(bs, -1, 2)  # expand -> [batch, npoints, 2]

        return points

    def create_edt2(self, points):
        edt2 = point_edt2(points, self.grid)

        return edt2