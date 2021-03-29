from unittest import TestCase

import torch

from dsketch.raster.composite import softor, smoothmax, over_recursive

B = 16
T = 10
C = 3
H = 100
W = 200
D = 100


class TestComposite(TestCase):
    data2d_1ch = torch.rand(B, T, H, W)
    data2d_3ch = torch.rand(B, C, T, H, W)
    data3d_1ch = torch.rand(B, T, D, H, W)
    data3d_3ch = torch.rand(B, C, D, T, H, W)

    def test_comp_dims(self):
        for f in [softor, smoothmax, over_recursive]:  # over
            self.assertEqual(f(TestComposite.data2d_1ch).shape, (B, H, W))
            self.assertEqual(f(TestComposite.data2d_3ch, dim=2).shape, (B, C, H, W))
            self.assertEqual(f(TestComposite.data3d_1ch).shape, (B, D, H, W))
            self.assertEqual(f(TestComposite.data3d_3ch, dim=3).shape, (B, C, D, H, W))
