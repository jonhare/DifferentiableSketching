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
            _size = size[i] if isinstance(size, torch.Tensor) else size
            if _size > 0:
                if pcols is None:
                    c.fill(path.circle(points[i, 1], -points[i, 0], _size))
                else:
                    c.fill(path.circle(points[i, 1], -points[i, 0], _size), [color.rgb(*pcols[i])])


def draw_lines_to_canvas(c, lines: torch.Tensor, lw=1, lcols=None):
    if lcols is not None:
        lcols = lcols.detach().cpu()

    if lines is not None:
        lines = lines.detach().cpu()
        for i in range(lines.shape[0]):
            _lw = lw[i] if isinstance(lw, torch.Tensor) else lw
            if _lw > 0:
                if lcols is None:
                    c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]),
                             [style.linewidth(_lw), style.linecap.round])
                else:
                    c.stroke(path.line(lines[i, 0, 1], -lines[i, 0, 0], lines[i, 1, 1], -lines[i, 1, 0]),
                             [style.linewidth(_lw), style.linecap.round, color.rgb(*lcols[i])])


def draw_points_lines(points: torch.Tensor, lines: torch.Tensor, filename, size=1, lw=1, pcols=None, lcols=None):
    c = canvas.canvas()

    draw_points_to_canvas(c, points, size, pcols)
    draw_lines_to_canvas(c, lines, lw, lcols)

    c.writePDFfile(file=filename)


def draw_points_lines_crs(points: torch.Tensor, lines: torch.Tensor, crs: torch.Tensor, filename, size=1, lw=1, clw=1,
                          pcols=None, lcols=None, crscols=None):
    c = canvas.canvas()

    draw_points_to_canvas(c, points, size, pcols)
    draw_lines_to_canvas(c, lines, lw, lcols)
    draw_crs_to_canvas(c, crs, clw, crscols)

    c.writePDFfile(file=filename)


def draw_crs_to_canvas(c, crs: torch.Tensor, lw=1, lcols=None, alpha=0.5):
    # crs [n, nc, 2]; only draw nc>1<nc-1
    if crs is None:
        return

    n, nc, _ = crs.shape

    # Premultiplied power constant for the following tj() function.
    alpha = alpha / 2

    def tj(ti, pi, pj):
        return ((pi - pj) ** 2).sum(dim=-1) ** alpha + ti

    if lcols is not None:
        lcols = lcols.detach().cpu()

    if crs is not None:
        crs = crs.detach().cpu()

        for i in range(n):
            _lw = lw[i] if isinstance(lw, torch.Tensor) else lw
            if _lw > 1e-3:
                for j in range(nc - 4 + 1):
                    p0 = crs[i, j + 0]
                    p1 = crs[i, j + 1]
                    p2 = crs[i, j + 2]
                    p3 = crs[i, j + 3]

                    t0 = 0
                    t1 = tj(t0, p0, p1)
                    t2 = tj(t1, p1, p2)
                    t3 = tj(t2, p2, p3)

                    c1 = (t2 - t1) / (t2 - t0)
                    c2 = (t1 - t0) / (t2 - t0)
                    d1 = (t3 - t2) / (t3 - t1)
                    d2 = (t2 - t1) / (t3 - t1)

                    m1 = (t2 - t1) * (c1 * (p1 - p0) / (t1 - t0) + c2 * (p2 - p1) / (t2 - t1))
                    m2 = (t2 - t1) * (d1 * (p2 - p1) / (t2 - t1) + d2 * (p3 - p2) / (t3 - t2))

                    q0 = p1
                    q1 = p1 + m1 / 3
                    q2 = p2 - m2 / 3
                    q3 = p2

                    curve = path.curve(q0[1], -q0[0],
                                       q1[1], -q1[0],
                                       q2[1], -q2[0],
                                       q3[1], -q3[0])
                    if lcols is None:
                        c.stroke(curve, [style.linewidth(_lw), style.linecap.round])
                    else:
                        c.stroke(curve, [style.linewidth(_lw), style.linecap.round, color.rgb(*lcols[i])])
