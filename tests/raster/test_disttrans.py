from unittest import TestCase
import torch
from dsketch.raster.disttrans import point_edt2, line_edt2, curve_edt2_bruteforce, linear_bezier
from dsketch.raster.disttrans import centripetal_catmull_rom_spline, quadratic_bezier, cubic_bezier


def make_grid(*sizes):
    ls = []
    for m in sizes:
        ls.append(torch.linspace(0, m-1, m))
    grid = torch.meshgrid(*ls)
    grid = torch.stack(grid, dim=-1)
    return grid


bs = 4


class TestDisttrans(TestCase):

    def test_point_edt2_2d(self):
        grid = make_grid(100, 100)
        point = torch.tensor([[50., 50.]])
        point = torch.cat([point] * bs, dim=0)

        raster = point_edt2(point, grid)
        self.assertEqual(raster.shape, (bs, 100, 100))
        self.assertAlmostEqual(raster[0, 50, 50], 0.0)

    def test_point_edt2_2d_ch(self):
        grid = make_grid(100, 100)
        point = torch.tensor([[[50., 50.], [40., 40.]]])
        point = torch.cat([point] * bs, dim=0)

        raster = point_edt2(point, grid)
        self.assertEqual(raster.shape, (bs, 2, 100, 100))
        self.assertAlmostEqual(raster[0, 0, 50, 50], 0.0)
        self.assertAlmostEqual(raster[0, 1, 40, 40], 0.0)

    def test_point_edt2_2d_i_ch(self):
        grid = make_grid(100, 100)
        point = torch.tensor([[[[50., 50.], [40., 40.]], [[10, 10], [70, 90]]]])
        point = torch.cat([point] * bs, dim=0)

        raster = point_edt2(point, grid)
        self.assertEqual(raster.shape, (bs, 2, 2, 100, 100))
        self.assertAlmostEqual(raster[0, 0, 0, 50, 50], 0.0)
        self.assertAlmostEqual(raster[0, 0, 1, 40, 40], 0.0)
        self.assertAlmostEqual(raster[0, 1, 0, 10, 10], 0.0)
        self.assertAlmostEqual(raster[0, 1, 1, 70, 90], 0.0)

    def test_point_edt2_3d(self):
        grid = make_grid(50, 100, 100)
        point = torch.tensor([[25., 50., 50.]])
        point = torch.cat([point] * bs, dim=0)

        raster = point_edt2(point, grid)
        self.assertEqual(raster.shape, (bs, 50, 100, 100))
        self.assertAlmostEqual(raster[0, 25, 50, 50], 0.0)

    def test_line_edt2_2d(self):
        grid = make_grid(100, 100)
        line = torch.tensor([[[50., 40.], [50., 60]]])
        line = torch.cat([line] * bs, dim=0)

        raster = line_edt2(line, grid)
        self.assertEqual(raster.shape, (bs, 100, 100))
        for i in range(40, 60):
            self.assertAlmostEqual(raster[0, 50, i], 0.0)
        for i in range(40, 60):
            self.assertAlmostEqual(raster[0, 49, i], 1.0)

    def test_line_edt2_2d_ch(self):
        grid = make_grid(100, 100)
        line = torch.tensor([[[[50., 40.], [50., 60]], [[40., 50.], [60., 50]]]])
        line = torch.cat([line] * bs, dim=0)

        raster = line_edt2(line, grid)
        self.assertEqual(raster.shape, (bs, 2, 100, 100))
        for i in range(40, 60):
            self.assertAlmostEqual(raster[0, 0, 50, i], 0.0)
            self.assertAlmostEqual(raster[0, 1, i, 50], 0.0)

    def test_line_edt2_3d(self):
        grid = make_grid(100, 100, 100)
        line = torch.tensor([[
            [[0, 50., 40.], [0, 50., 60]],
            [[0, 40., 50.], [0, 60., 50]],
            [[40, 50, 50.], [60, 50., 50]]]])
        line = torch.cat([line] * bs, dim=0)

        raster = line_edt2(line, grid)
        self.assertEqual(raster.shape, (bs, 3, 100, 100, 100))
        for i in range(40, 60):
            self.assertAlmostEqual(raster[0, 0, 0, 50, i], 0.0)
            self.assertAlmostEqual(raster[0, 1, 0, i, 50], 0.0)
            self.assertAlmostEqual(raster[0, 2, i, 50, 50], 0.0)

    def test_linear_curve_edt2_bruteforce_2d(self):
        grid = make_grid(100, 100)
        line = torch.tensor([[[50., 40.], [50., 60]]])
        line = torch.cat([line] * bs, dim=0)

        raster = curve_edt2_bruteforce(line, grid, iters=5, slices=20, cfcn=linear_bezier)
        self.assertEqual(raster.shape, (bs, 100, 100))
        for i in range(40, 60):
            self.assertAlmostEqual(raster[0, 50, i].item(), 0.0)
        for i in range(40, 60):
            self.assertAlmostEqual(raster[0, 49, i].item(), 1.0)

    def test_linear_curve_edt2_bruteforce_3d(self):
        grid = make_grid(10, 10, 10)
        line = torch.tensor([[[4., 5., 5], [6., 5, 5]]])
        line = torch.cat([line] * bs, dim=0)

        raster = curve_edt2_bruteforce(line, grid, iters=1, slices=10, cfcn=linear_bezier)
        self.assertEqual(raster.shape, (bs, 10, 10, 10))
        for i in range(4, 6):
            self.assertAlmostEqual(raster[0, i, 5, 5].item(), 0.0, delta=0.001)

    def test_curve_edt2_bruteforce_2d(self):
        grid = make_grid(100, 100)
        line = torch.tensor([[[50., 30.], [50., 40.], [50., 60.], [50., 70]]])
        line = torch.cat([line] * bs, dim=0)

        for f in [cubic_bezier, quadratic_bezier, centripetal_catmull_rom_spline]:
            raster = curve_edt2_bruteforce(line, grid, iters=5, slices=20, cfcn=f)
            self.assertEqual(raster.shape, (bs, 100, 100))

    def test_curve_edt2_bruteforce_3d(self):
        grid = make_grid(10, 10, 10)
        line = torch.tensor([[[5., 3., 5.], [5., 40., 5.], [5., 6., 5.], [5., 7., 5.]]])
        line = torch.cat([line] * bs, dim=0)

        for f in [cubic_bezier, quadratic_bezier, centripetal_catmull_rom_spline]:
            raster = curve_edt2_bruteforce(line, grid, iters=5, slices=20, cfcn=f)
            self.assertEqual(raster.shape, (bs, 10, 10, 10))

