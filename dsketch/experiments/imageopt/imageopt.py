import argparse
import importlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from dsketch.experiments.shared.args_losses import loss_choices, get_loss
from dsketch.raster.composite import softor, over
from dsketch.raster.disttrans import point_edt2, line_edt2, curve_edt2_polyline, catmull_rom_spline


# from dsketch.raster.raster import exp


def exp(dt2: torch.Tensor, sigma2) -> torch.Tensor:
    if type(sigma2) != torch.Tensor:
        return torch.exp(-1 * dt2 / sigma2)

    tmp = -1 * dt2
    tmp2 = []
    for i in range(tmp.shape[0]):
        tmp2.append(tmp[i, :, :] / sigma2[i])
    return torch.exp(torch.stack(tmp2, dim=0))


def save_image(img, fp):
    img = img.squeeze(0).detach()  # remove batch dim

    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    if img.shape[0] == 1:
        img = 1 - img  # always invert so it's black lines on white
        img2 = torch.cat([img] * 3, dim=0)
    else:
        img2 = 1 - img  # always invert so it's black lines on white
        # img2[2, :, :] = 1 - img2[2, :, :]  # always invert so it's black lines on white

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = img2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # if img.shape[0] == 1:
    im = Image.fromarray(ndarr)
    # else:
    #     im = Image.fromarray(ndarr, "HSV").convert("RGB")
    im.save(fp, format=None)


def save_pdf(params, cparams, args, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    params = params.detach().cpu()
    if cparams is not None:
        cparams = cparams.detach().cpu()
        cparams = 1 - cparams
        # cparams[:, 2] = 1 - cparams[:, 2]
        # ndarr = cparams.unsqueeze(0).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        # cparams = torch.from_numpy(np.array(Image.fromarray(ndarr, "HSV").convert("RGB"), dtype=np.float32) / 255.)[0]

    pparams = None
    cpparams = None
    lparams = None
    clparams = None
    crsparams = None
    ccrsparams = None

    lw = math.sqrt(args.sigma2_current / args.sf) / 0.54925
    # lw = 2

    if args.points > 0:
        pparams = params[0:2 * args.points].view(-1, 2)
        pparams[:, 0] = args.target_shape[-2] * (pparams[:, 0] + args.grid_row_extent) / (2 * args.grid_row_extent)
        pparams[:, 1] = args.target_shape[-1] * (pparams[:, 1] + args.grid_col_extent) / (2 * args.grid_col_extent)

        if cparams is not None:
            cpparams = cparams[:args.points]

    if args.lines > 0:
        lparams = params[2 * args.points:2 * args.points + 4 * args.lines].view(-1, 2, 2)
        lparams[:, 0, 0] = args.target_shape[-2] * (lparams[:, 0, 0] + args.grid_row_extent) / (
                2 * args.grid_row_extent)
        lparams[:, 0, 1] = args.target_shape[-1] * (lparams[:, 0, 1] + args.grid_col_extent) / (
                2 * args.grid_col_extent)
        lparams[:, 1, 0] = args.target_shape[-2] * (lparams[:, 1, 0] + args.grid_row_extent) / (
                2 * args.grid_row_extent)
        lparams[:, 1, 1] = args.target_shape[-1] * (lparams[:, 1, 1] + args.grid_col_extent) / (
                2 * args.grid_col_extent)

        if cparams is not None:
            clparams = cparams[args.points:args.points + args.lines]

    if args.crs > 0:
        crsparams = params[2 * args.points + 4 * args.lines:].view(args.crs, 2 + args.crs_points, 2)
        crsparams[:, :, 0] = args.target_shape[-2] * (crsparams[:, :, 0] + args.grid_row_extent) / (
                2 * args.grid_row_extent)
        crsparams[:, :, 1] = args.target_shape[-1] * (crsparams[:, :, 1] + args.grid_col_extent) / (
                2 * args.grid_col_extent)

        if cparams is not None:
            ccrsparams = cparams[args.points + args.lines:]

    # draw_points_lines_crs(pparams, lparams, crsparams, file, lw=lw, clw=lw, pcols=cpparams, lcols=clparams,
    #                       crscols=ccrsparams)


def make_optimiser(args, params, cparams=None, sigma2params=None):
    module = importlib.import_module('torch.optim')
    opt = getattr(module, args.optimiser)
    p = [{'params': params, 'lr': args.lr}]
    if cparams is not None:
        lr = args.colour_lr if 'colour_lr' in args else args.lr
        p.append({'params': cparams, 'lr': lr})
    if sigma2params is not None:
        lr = args.sigma2_lr if 'sigma2_lr' in args else args.lr
        p.append({'params': sigma2params, 'lr': lr, 'betas': (0, 0)})
    return opt(p)


def optimise(target, params, cparams, sigma2params, render_fn, args):
    params.requires_grad = True
    if cparams is not None:
        cparams.requires_grad = True

    if sigma2params is not None:
        sigma2params.requires_grad = True
        sigma2 = sigma2params
    else:
        sigma2 = args.init_sigma2

    loss_fn = get_loss(args.loss)(args)
    optim = make_optimiser(args, params, cparams, sigma2params)

    itr = tqdm(range(args.iters))
    for i in itr:
        optim.zero_grad()

        est = render_fn(params, cparams, sigma2)
        lss = loss_fn(est, target)
        lss.backward()
        optim.step()
        params = clamp_params(params, args)
        if cparams is not None:
            clamp_colour_params(cparams)

        if sigma2params is not None:
            mask = sigma2params.data < 1e-6

            if args.crs > 0 and args.restarts:
                # TODO: same for pts and lines
                crsparams = params[2 * args.points + 4 * args.lines:].view(args.crs, 2 + args.crs_points, 2).data

                for j in range(len(mask)):
                    jj = args.points + args.lines + j
                    if mask[j] and i < args.iters / 2:
                        crsparams[jj] = torch.rand_like(crsparams[j]) - 0.5
                        crsparams[jj, :, 0] *= 2 * args.grid_row_extent
                        crsparams[jj, :, 1] *= 2 * args.grid_col_extent
                        crsparams[jj, -2, 0] = crsparams[jj, 1, 0] + 0.3 * crsparams[jj, -2, 0]
                        crsparams[jj, -2, 1] = crsparams[jj, 1, 1] + 0.3 * crsparams[jj, -2, 1]
                        sigma2params.data[j] += args.init_sigma2

            if i < args.iters / 2 and args.restarts:
                sigma2params.data.clamp_(1e-6, args.init_sigma2)
            else:
                sigma2params.data.clamp_(1e-10, args.init_sigma2)

        if sigma2params is None:
            if i % args.sigma2_step == 0:
                sigma2 = sigma2 * args.sigma2_factor
                if sigma2 < args.final_sigma2:
                    sigma2 = args.final_sigma2

            args.sigma2_current = sigma2
            itr.set_postfix({'loss': lss.item(), 'sigma^2': sigma2})
        else:
            itr.set_postfix({'loss': lss.item(), 'sigma^2': 'learned'})

        if args.snapshots_path is not None and i % args.snapshots_steps == 0:
            ras = render_fn(params, cparams, sigma2)
            save_image(ras.detach().cpu(), args.snapshots_path + "/snapshot_" + str(i) + ".png")
            save_pdf(params, cparams, args, args.snapshots_path + "/snapshot_" + str(i) + ".pdf")

    return params


def render_points(params, sigma2, grid):
    return exp(point_edt2(params, grid), sigma2).unsqueeze(0)


def render_lines(params, sigma2, grid):
    return exp(line_edt2(params, grid), sigma2).unsqueeze(0)


def render_crs(params, sigma2, grid, coordpairs):
    ncrs = params.shape[0]
    crs = torch.cat((params[:, coordpairs[:, 0]],
                     params[:, coordpairs[:, 1]],
                     params[:, coordpairs[:, 2]],
                     params[:, coordpairs[:, 3]]), dim=-1)  # [batch, nlines, 8]

    crs = crs.view(ncrs, -1, 4, 2)

    return softor(exp(curve_edt2_polyline(crs, grid, 10, cfcn=catmull_rom_spline), sigma2), dim=1).unsqueeze(0)


def clamp_params(params, args):
    if args.points > 0:
        pparams = params[0:2 * args.points].view(args.points, 2).data
        pparams[:, 0].clamp_(-args.grid_row_extent, args.grid_row_extent)
        pparams[:, 1].clamp_(-args.grid_col_extent, args.grid_col_extent)

    if args.lines > 0:
        lparams = params[2 * args.points: 2 * args.points + 4 * args.lines].view(args.lines, 2, 2).data
        lparams[:, 0, 0].clamp_(-args.grid_row_extent, args.grid_row_extent)
        lparams[:, 1, 0].clamp_(-args.grid_row_extent, args.grid_row_extent)
        lparams[:, 0, 1].clamp_(-args.grid_col_extent, args.grid_col_extent)
        lparams[:, 1, 1].clamp_(-args.grid_col_extent, args.grid_col_extent)

    if args.crs > 0:
        crsparams = params[2 * args.points + 4 * args.lines:].view(args.crs, 2 + args.crs_points, 2).data
        crsparams[:, 1:-2, 0].clamp_(-args.grid_row_extent, args.grid_row_extent)
        crsparams[:, 1:-2, 1].clamp_(-args.grid_col_extent, args.grid_col_extent)

    return params


def clamp_colour_params(params):
    # params[params[:, 0] < 0, 0] += 1
    # params[params[:, 0] > 1, 0] -= 1
    params.data.clamp_(0, 1)


def render(params, cparams, sigma2, grid, coordpairs, args):
    ras = []

    if args.points > 0:
        pparams = params[0:2 * args.points].view(args.points, 2)
        if type(sigma2) != torch.Tensor:
            pts = render_points(pparams, sigma2, grid)
        else:
            pts = render_points(pparams, sigma2[0:args.points], grid)
        ras.append(pts)

    if args.lines > 0:
        lparams = params[2 * args.points: 2 * args.points + 4 * args.lines].view(args.lines, 2, 2)
        if type(sigma2) != torch.Tensor:
            lns = render_lines(lparams, sigma2, grid)
        else:
            lns = render_lines(lparams, sigma2[args.points:args.points + args.lines], grid)
        ras.append(lns)

    if args.crs > 0:
        crsparams = params[2 * args.points + 4 * args.lines:].view(args.crs, 2 + args.crs_points, 2)
        if type(sigma2) != torch.Tensor:
            crs = render_crs(crsparams, sigma2, grid, coordpairs)
        else:
            crs = render_crs(crsparams, sigma2[args.points + args.lines:], grid, coordpairs)
        ras.append(crs)

    ras = torch.cat(ras, dim=0)  # [1, nprim, row, col]

    if cparams is not None:
        ras = ras.unsqueeze(2)  # [1, nprim, 1, row, col]
        ras = ras.repeat_interleave(3, dim=2)  # [1, nprim, 3, row, col]
        lab = cparams.unsqueeze(-1).unsqueeze(-1)  # npts, 4, 1, 1
        ras = lab * ras
        return over(ras, dim=1, keepdim=False)
        # return over_recursive(ras, dim=1)

    return softor(ras, dim=1, keepdim=True)


def make_init_params(args, img):
    torch.random.manual_seed(args.seed)

    pparams = torch.rand((args.points, 2), device=args.device)
    pparams[:, 0] = 2 * (pparams[:, 0] - 0.5) * args.grid_row_extent
    pparams[:, 1] = 2 * (pparams[:, 1] - 0.5) * args.grid_col_extent
    pparams = pparams.view(-1)

    lparams = torch.rand((args.lines, 2, 2), device=args.device)
    lparams[:, 0, 0] -= 0.5
    lparams[:, 0, 1] -= 0.5
    lparams[:, 0, 0] *= 2 * args.grid_row_extent
    lparams[:, 0, 1] *= 2 * args.grid_col_extent
    lparams[:, 1, 0] = lparams[:, 0, 0] + 0.2 * (lparams[:, 1, 0] - 0.5)
    lparams[:, 1, 1] = lparams[:, 0, 1] + 0.2 * (lparams[:, 1, 1] - 0.5)
    lparams = lparams.view(-1)

    assert args.crs_points >= 2, "must be at least two crs-points"
    crsparams = torch.rand((args.crs, 2 + args.crs_points, 2), device=args.device)
    crsparams[:, :, 0] -= 0.5
    crsparams[:, :, 1] -= 0.5
    crsparams[:, :, 0] *= 2 * args.grid_row_extent
    crsparams[:, :, 1] *= 2 * args.grid_col_extent
    # crsparams[:, 3, 0] = crsparams[:, 0, 0] + 0.5 * (crsparams[:, 3, 0] - 0.5)
    # crsparams[:, 3, 1] = crsparams[:, 0, 1] + 0.5 * (crsparams[:, 3, 1] - 0.5)
    #
    # crsparams[:, 1, 0] = (crsparams[:, 1, 0] - 0.5) * 0.2 + crsparams[:, 0, 0]
    # crsparams[:, 1, 1] = (crsparams[:, 1, 1] - 0.5) * 0.2 + crsparams[:, 0, 1]
    # crsparams[:, 2, 0] = (crsparams[:, 2, 0] - 0.5) * 0.2 + crsparams[:, 3, 0]
    # crsparams[:, 2, 1] = (crsparams[:, 2, 1] - 0.5) * 0.2 + crsparams[:, 3, 1]
    crsparams[:, -2, 0] = crsparams[:, 1, 0] + 0.3 * crsparams[:, -2, 0]
    crsparams[:, -2, 1] = crsparams[:, 1, 1] + 0.3 * crsparams[:, -2, 1]

    crsparams = crsparams.view(-1)

    return clamp_params(torch.cat((pparams, lparams, crsparams), dim=0), args)


def add_shared_args(parser):
    parser.add_argument("image", help="path to image", type=str)
    parser.add_argument("--width", type=int,
                        help="Width to scale input to for optimisation (aspect ratio is preserved).")
    parser.add_argument("--lines", type=int, required=False, default=0, help="number of line segments.")
    parser.add_argument("--points", type=int, required=False, default=0, help="number of points.")
    parser.add_argument("--loss", choices=loss_choices(), required=True, help="loss function")
    parser.add_argument("--iters", type=int, required=False, help="number of iterations.", default=8000)
    parser.add_argument("--init-sigma2", type=float, required=False, help="initial sigma^2.", default=0.55 ** 2)
    parser.add_argument("--final-sigma2", type=float, required=False, help="final sigma^2.", default=0.55 ** 2)
    parser.add_argument("--sigma2-factor", type=float, required=False,
                        help="factor to multiply sigma^2 by every sigma2-step.", default=0.5)
    parser.add_argument("--sigma2-step", type=int, required=False,
                        help="number of iterations between changes in sigma^2", default=100)
    parser.add_argument("--seed", type=int, required=False, help="seed for initial params", default=1)

    parser.add_argument("--lr", type=float, required=False, help="learning rate", default=1e-2)

    parser.add_argument("--target-raster", type=str, required=False, help="path to save target raster")

    parser.add_argument("--init-raster", type=str, required=False, help="path to save initial raster")
    parser.add_argument("--init-pdf", type=str, required=False, help="path to save initial pdf")

    parser.add_argument("--final-raster", type=str, required=False, help="path to save final raster")
    parser.add_argument("--final-pdf", type=str, required=False, help="path to save final pdf")

    parser.add_argument("--snapshots-path", type=str, required=False, help="path to save snapshots")
    parser.add_argument("--snapshots-steps", type=int, required=False, help="snapshots interval",
                        default=1000)

    parser.add_argument("--invert", action='store_true', required=False, help="should that target image be inverted?")
    parser.add_argument("--optimiser", type=str, required=False, help="torch.optim class to use for optimisation",
                        default='Adam')
    parser.add_argument("--device", help='device to use', required=False, type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--colour", action='store_true', required=False, help="optimise a colour image")
    parser.add_argument("--crs", type=int, required=False, help="number of catmull-rom splines", default=0)
    parser.add_argument("--crs-points", type=int, required=False,
                        help="number of catmull-rom points (excluding end control points", default=2)
    parser.add_argument("--opt-sigma2", action='store_true', required=False, help="optimise widths")
    parser.add_argument("--sigma2-lr", type=float, required=False,
                        help="sigma2 learning rate (defaults to --lr if not set)")
    parser.add_argument("--colour-lr", type=float, required=False,
                        help="colour learning rate (defaults to --lr if not set)")
    parser.add_argument("--restarts", action='store_true', required=False, default=False,
                        help="reinit params if sigma2 becomes too small")


def main():
    fake_parser = argparse.ArgumentParser(add_help=False)
    add_shared_args(fake_parser)

    fake_args, _ = fake_parser.parse_known_args()

    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    get_loss(fake_args.loss).add_args(parser)

    args = parser.parse_args()

    target = Image.open(args.image)

    if args.width:
        basewidth = args.width
        wpercent = (basewidth / float(target.size[0]))
        hsize = int((float(target.size[1]) * float(wpercent)))
        target = target.resize((basewidth, hsize), Image.ANTIALIAS)

    if args.colour:
        args.channels = 3
        # target = np.array(target.convert("HSV"), dtype=np.float32) / 255.
        target = np.array(target.convert("RGB"), dtype=np.float32) / 255.
        target = torch.from_numpy(target).to(args.device).unsqueeze(0).permute(0, 3, 1, 2)  # 1CHW
        cparams = torch.rand((args.points + args.lines + args.crs, 3), device=args.device)

        target = 1 - target
        # target[:, 2, :, :] = 1 - target[:, 2, :, :]  # invert the brightness channel
    else:
        args.channels = 1
        target = np.array(target.convert("L"), dtype=np.float32) / 255.
        target = torch.from_numpy(target).to(args.device).unsqueeze(0).unsqueeze(0)  # 1CHW
        cparams = None

        if args.invert:
            target = 1 - target

    if args.target_raster is not None:
        save_image(target, args.target_raster)

    args.target_shape = target.shape
    args.grid_row_extent = 1
    args.grid_col_extent = target.shape[-1] / target.shape[-2]

    r = torch.linspace(-args.grid_row_extent, args.grid_row_extent, target.shape[-2])
    c = torch.linspace(-args.grid_col_extent, args.grid_col_extent, target.shape[-1])
    grid = torch.meshgrid(r, c)
    grid = torch.stack(grid, dim=2).to(args.device)

    # scale the sigmas to match the grid defined above, rather than being relative to 1 pixel
    args.sf = (2 / target.shape[-2]) ** 2
    args.init_sigma2 = args.init_sigma2 * args.sf
    args.final_sigma2 = args.final_sigma2 * args.sf
    args.sigma2_current = args.init_sigma2

    sigma2params = None
    if args.opt_sigma2:
        sigma2params = torch.ones(args.points + args.lines + args.crs,
                                  device=args.device) * args.sigma2_current

    params = make_init_params(args, target)

    # pairs for crs splines
    coordpairs = torch.stack([torch.arange(0, args.crs_points + 2 - 3, 1),
                              torch.arange(1, args.crs_points + 2 - 2, 1),
                              torch.arange(2, args.crs_points + 2 - 1, 1),
                              torch.arange(3, args.crs_points + 2, 1)], dim=1)

    def r(p, cp, s):
        return render(p, cp, s, grid, coordpairs, args)

    if args.init_raster is not None:
        ras = r(params, cparams, args.final_sigma2)
        save_image(ras.detach().cpu(), args.init_raster)

    if args.init_pdf is not None:
        save_pdf(params, cparams, args, args.init_pdf)

    params = optimise(target, params, cparams, sigma2params, r, args)

    if args.final_raster is not None:
        ras = r(params, cparams, args.final_sigma2)
        save_image(ras.detach().cpu(), args.final_raster)

    if args.final_pdf is not None:
        save_pdf(params, cparams, args, args.final_pdf)
