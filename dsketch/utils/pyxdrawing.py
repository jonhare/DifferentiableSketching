"""
Some utility functions to save vector images from point and line parameters
"""

import torch
from pyx import canvas, path, style, color


def draw_line_segments(lines: torch.Tensor, filename, lw=1):
    c = canvas.canvas()

    lines = lines.detach().cpu()
    lw = style.linewidth(lw)

    for i in range(lines.shape[0]):
        c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]), [lw, style.linecap.round])

    c.writePDFfile(file=filename)


def draw_points(points: torch.Tensor, filename, size=1):
    c = canvas.canvas()

    points = points.detach().cpu()

    for i in range(points.shape[0]):
        c.fill(path.circle(points[i, 1], -points[i, 0], size))

    c.writePDFfile(file=filename)


def draw_points_to_canvas(c, points: torch.Tensor, size=1, pcols=None):
    if pcols is not None:
        pcols = pcols.detach().cpu()

    if points is not None:
        points = points.detach().cpu()
        for i in range(points.shape[0]):
            if pcols is None:
                c.fill(path.circle(points[i, 1], -points[i, 0], size))
            else:
                c.fill(path.circle(points[i, 1], -points[i, 0], size), [color.rgb(*pcols[i])])


def draw_lines_to_canvas(c, lines: torch.Tensor, lw=1, lcols=None):
    if lcols is not None:
        lcols = lcols.detach().cpu()

    if lines is not None:
        lines = lines.detach().cpu()
        lw = style.linewidth(lw)
        for i in range(lines.shape[0]):
            if lcols is None:
                c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]),
                         [lw, style.linecap.round])
            else:
                c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]),
                         [lw, style.linecap.round, color.rgb(*lcols[i])])


def draw_points_lines(points: torch.Tensor, lines: torch.Tensor, filename, size=1, lw=1, pcols=None, lcols=None):
    c = canvas.canvas()

    draw_points_to_canvas(c, points, size, pcols)
    draw_lines_to_canvas(c, lines, lw, lcols)

    c.writePDFfile(file=filename)


def draw_crs_to_canvas(c, crs: torch.Tensor, lw=1, lcols=None):
    # crs [n, nc, 2]; only draw nc>1<nc-1
    if lcols is not None:
        lcols = lcols.detach().cpu()

    c.stroke
    # if lines is not None:
    #     lines = lines.detach().cpu()
    #     lw = style.linewidth(lw)
    #     for i in range(lines.shape[0]):
    #         if lcols is None:
    #             c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]),
    #                      [lw, style.linecap.round])
    #         else:
    #             c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]),
    #                      [lw, style.linecap.round, color.rgb(*lcols[i])])
