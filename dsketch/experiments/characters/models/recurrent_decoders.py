from abc import ABC

import torch
import torch.nn as nn

from dsketch.experiments.characters.models.model_bases import Decoder, AdjustableSigmaMixin, get_model
from dsketch.experiments.characters.models.single_pass_decoders import SinglePassSimpleLineDecoder, \
    SinglePassPolyConnectDecoder, SinglePassCRSDecoder, SinglePassPolyLineDecoder
from dsketch.experiments.shared import metrics
from dsketch.raster.composite import softor
from dsketch.raster.disttrans import curve_edt2_polyline
from dsketch.raster.raster import exp


class RecurrentDecoder(Decoder, ABC):
    def forward(self, inp, state=None):
        bs = inp.shape[0]
        composite = torch.zeros(bs, 1, self.sz, self.sz, device=inp.device)

        for i in range(self.steps):
            prev_latent = self.encoder(composite)
            latent = torch.cat((inp, prev_latent), dim=-1)

            params = self.decode_to_params(latent)
            edt2 = self.create_edt2(params)
            sigma2 = self.get_sigma2(params)
            ras = self.raster_soft(edt2, sigma2)

            composite = softor(torch.cat((composite, ras), dim=1), keepdim=True)

        if state is not None:
            hr = self.raster_hard(edt2)
            if metrics.HARDRASTER in state:
                state[metrics.HARDRASTER] = softor(torch.stack((state[metrics.HARDRASTER], hr), dim=1), keepdim=False)
                state[metrics.SQ_DISTANCE_TRANSFORM], _ = torch.min(torch.stack((state[metrics.HARDRASTER], hr), dim=1),
                                                                    dim=1)
            else:
                state[metrics.HARDRASTER] = hr
                state[metrics.SQ_DISTANCE_TRANSFORM] = edt2

        return composite * self.args.contrast

    def decode_to_params(self, inp):
        return self.rasteriser.decode_to_params(inp)

    def create_edt2(self, lines):
        return self.rasteriser.create_edt2(lines)


class RecurrentSimpleLineDecoder(AdjustableSigmaMixin, RecurrentDecoder):
    def __init__(self, args, enc, steps=5, nlines=5, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2):
        super().__init__(args)

        self.steps = steps
        self.sigma2 = sigma2
        self.sz = sz
        self.rasteriser = SinglePassSimpleLineDecoder(args, nlines, input * 2, hidden, hidden2, sz, sigma2)
        self.grid = self.rasteriser.grid
        self.encoder = enc

    @staticmethod
    def _add_args(p):
        p.add_argument("--steps", help="number of steps for recurrent models", type=int, default=5, required=False)
        SinglePassSimpleLineDecoder._add_args(p)

    @staticmethod
    def create(args):
        enc = get_model(args.encoder).create(args)
        return RecurrentSimpleLineDecoder(args, enc, steps=args.steps, nlines=args.nlines, input=args.latent_size,
                                          hidden=args.decoder_hidden,
                                          hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2)


class RecurrentPolyLineDecoder(AdjustableSigmaMixin, RecurrentDecoder):
    def __init__(self, args, enc, steps=5, npoints=8, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2):
        super().__init__(args)

        self.steps = steps
        self.sigma2 = sigma2
        self.sz = sz
        self.rasteriser = SinglePassPolyLineDecoder(args, npoints, input * 2, hidden, hidden2, sz, sigma2)
        self.grid = self.rasteriser.grid
        self.encoder = enc

    @staticmethod
    def _add_args(p):
        p.add_argument("--steps", help="number of steps for recurrent models", type=int, default=5, required=False)
        SinglePassPolyLineDecoder._add_args(p)

    @staticmethod
    def create(args):
        enc = get_model(args.encoder).create(args)
        return RecurrentPolyLineDecoder(args, enc, steps=args.steps, npoints=args.npoints, input=args.latent_size,
                                        hidden=args.decoder_hidden,
                                        hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2)


class RecurrentPolyConnectDecoder(AdjustableSigmaMixin, RecurrentDecoder):
    def __init__(self, args, enc, steps=5, npoints=8, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2,
                 allow_points=True):
        super().__init__(args)

        self.steps = steps
        self.sigma2 = sigma2
        self.sz = sz
        self.rasteriser = SinglePassPolyConnectDecoder(args, npoints, input * 2, hidden, hidden2, sz, sigma2,
                                                       allow_points)
        self.grid = self.rasteriser.grid
        self.encoder = enc

    @staticmethod
    def _add_args(p):
        p.add_argument("--steps", help="number of steps for recurrent models", type=int, default=5, required=False)
        SinglePassPolyConnectDecoder._add_args(p)

    @staticmethod
    def create(args):
        enc = get_model(args.encoder).create(args)
        return RecurrentPolyConnectDecoder(args, enc, steps=args.steps, npoints=args.npoints, input=args.latent_size,
                                           hidden=args.decoder_hidden, hidden2=args.decoder_hidden2, sz=args.size,
                                           sigma2=args.sigma2, allow_points=args.allowPoints)

    def decode_to_params(self, inp):
        lines, conn = self.rasteriser.decode_to_params(inp)

        return lines, conn

    def create_edt2(self, lines):
        edt2 = self.rasteriser.create_edt2(lines)

        return edt2

    def raster_soft(self, edt2, conn):
        rasters = exp(edt2, self.sigma2)
        rasters = rasters * conn.view(edt2.shape[0], -1, 1, 1)
        return softor(rasters, keepdim=True)

    def forward(self, inp, state=None):
        bs = inp.shape[0]
        composite = torch.zeros(bs, 1, self.sz, self.sz, device=inp.device)
        lines = []

        edt2 = None
        for i in range(self.steps):
            prev_latent = self.encoder(composite)
            latent = torch.cat((inp, prev_latent), dim=-1)

            params, conn = self.decode_to_params(latent)

            lines.append(params)  # is this correct here?

            edt2 = self.create_edt2(params)
            ras = self.raster_soft(edt2, conn)

            composite = softor(torch.cat((composite, ras), dim=1)).unsqueeze(1)

        if state is not None:
            state[metrics.HARDRASTER] = self.raster_hard(edt2)
            state[metrics.SQ_DISTANCE_TRANSFORM] = edt2

        return composite


class RecurrentCRSDecoder(AdjustableSigmaMixin, RecurrentDecoder):
    def __init__(self, args, enc, steps=5, npoints=8, nlines=1, input=64, hidden=64, hidden2=256, sz=28, sigma2=1e-2,
                 edt_approx='polyline'):
        super().__init__(args)

        self.steps = steps
        self.sigma2 = sigma2
        self.sz = sz
        self.rasteriser = SinglePassCRSDecoder(args, npoints, nlines, input * 2, hidden, hidden2, sz, edt_approx,
                                               sigma2)
        self.grid = self.rasteriser.grid
        self.encoder = enc

    @staticmethod
    def _add_args(p):
        p.add_argument("--steps", help="number of steps for recurrent models", type=int, default=5, required=False)
        SinglePassCRSDecoder._add_args(p)

    @staticmethod
    def create(args):
        enc = get_model(args.encoder).create(args)
        return RecurrentCRSDecoder(args, enc, steps=args.steps, npoints=args.npoints, nlines=args.nlines,
                                   input=args.latent_size, hidden=args.decoder_hidden,
                                   hidden2=args.decoder_hidden2, sz=args.size, sigma2=args.sigma2,
                                   edt_approx=args.edt_approximation)


class RNNBezierDecoder(AdjustableSigmaMixin, Decoder):
    class DecoderRNN(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            output_size = 8  # 4x 2d points
            self.hidden_size = hidden_size

            self.embedding = nn.Linear(output_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)

        def forward(self, input, hidden):
            output = self.embedding(input)
            output = torch.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.out(output)
            return output, hidden

    def __init__(self, args, latent_size, steps, sz, sigma2):
        super().__init__(args)
        self.steps = steps
        self.sigma2 = sigma2
        self.rnn = RNNBezierDecoder.DecoderRNN(latent_size)
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

    @staticmethod
    def _add_args(p):
        p.add_argument("--steps", help="number of steps for recurrent models", type=int, default=5, required=False)
        AdjustableSigmaMixin._add_args(p)

    def create_edt2(self, params):
        return curve_edt2_polyline(params, self.grid, 10)

    @staticmethod
    def create(args):
        return RNNBezierDecoder(args, args.latent_size, args.steps, args.size, args.sigma2)

    def decode_to_params(self, inp):
        decoder_input = torch.zeros((1, inp.shape[0], 8), device=inp.device)
        decoder_hidden = inp.unsqueeze(0)
        decoded = []
        for i in range(self.steps):
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            decoded.append(decoder_output.permute(1, 0, 2).view(inp.shape[0], 1, 4, 2))
            decoder_input = decoder_output.detach()
        return torch.cat(decoded, dim=1)
