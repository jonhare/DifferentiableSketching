from typing import Callable, Optional

import torch
from quickdraw import QuickDrawData, QuickDrawing, QuickDrawDataGroup
from torch.types import Number
from torch.utils.data.dataset import Dataset

from ..raster.composite import softor
from ..raster.disttrans import line_edt2
from ..raster.raster import exp, nearest_neighbour, compute_nearest_neighbour_sigma_bres

from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor


# pytorch dataset with all Quickdraw classes, with default of 1000 samples per class
class QuickDrawDataset(Dataset):
    def __init__(self,
                 recognized: Optional[bool] = None,
                 transform: Callable[[QuickDrawing], torch.Tensor] = None):

        self.qd = QuickDrawData()
        self.qd_class_names = self.qd.drawing_names
        
        # dictionary of QuickDrawDataGroups based on all possible names, loads 1000 examples from each class, but can
        # be changed by specifying max_drawings
        self.qd_DataGroups = {name: QuickDrawDataGroup(name, recognized=recognized) for name in self.qd_class_names}
        
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __getitem__(self, index):
        class_index = index//1000
        insideclass_index = index%1000
        datagroup = self.qd_DataGroups[self.qd_class_names[class_index]]
        return self.transform(datagroup.get_drawing(insideclass_index))

    def __len__(self):
        return len(self.qd_class_names)*1000


# pytorch dataset for a quickdraw data group; uses a 'transform' to convert the
# QuickDrawing object into something more useful
class QuickDrawDataGroupDataset(Dataset):
    def __init__(self,
                 name: str,
                 max_drawings: int = 1000,
                 recognized: Optional[bool] = None,
                 transform: Callable[[QuickDrawing], torch.Tensor] = None):

        self.ds = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.ds.get_drawing(index))

    def __len__(self):
        return self.ds.drawing_count


def get_line_segments(qd: QuickDrawing) -> torch.Tensor:
    """Convert a QuickDrawing to a tensor of line segment parameters.

    Returned coordinates are in the default [0..256, 0..256] range used by QuickDraw.

    Args:
        qd: the sketch data to convert

    Returns:
        A tensor of shape [N, 2, 2] where N is the number of line segments and each row contains
        [[start_i, start_j], [end_i, end_j]].
    """
    pts = []
    for stroke in qd.strokes:
        start = stroke[0]
        for i in range(1, len(stroke)):
            end = stroke[i]

            pts.append([[start[1], start[0]], [end[1], end[0]]])  # swap x,y to i,j
            start = end

    params = torch.tensor(pts, dtype=torch.get_default_dtype())

    return params


# class to rasterise a QuickDraw image using the dsketch machinery
class QuickDrawRasterise:
    def __init__(self, hard: bool = True, sigma: Number = None, device=None):
        a = torch.linspace(0, 255, 256)
        grid = torch.meshgrid(a, a)
        self.grid = torch.stack(grid, dim=2)

        self.hard = hard

        self.device = device

        if sigma is None and hard is False:
            raise ValueError("Sigma must be set for soft rasterisation")

        if sigma is None and hard is True:
            sigma = compute_nearest_neighbour_sigma_bres(self.grid)

        self.sigma = sigma

    def __call__(self, qd: QuickDrawing) -> torch.Tensor:
        params = get_line_segments(qd).to(self.device)
        edts = line_edt2(params, self.grid)
        if self.hard:
            ras = nearest_neighbour(edts, self.sigma)
        else:
            ras = exp(edts, self.sigma)

        # Render image (compositions work on a batch, so adding extra dim); this extra dim becomes the channel dim
        img = softor(ras.unsqueeze(0))

        return img


# class to rasterise a QuickDraw image using PIL
class QuickDrawRasterisePIL:
    def __init__(self, stroke_width=1):
        self.stroke_width = stroke_width

    def __call__(self, qd: QuickDrawing) -> torch.Tensor:
        image = Image.new("L", (256, 256), color=0)
        image_draw = ImageDraw.Draw(image)

        for stroke in qd.strokes:
            image_draw.line(stroke, fill=255, width=self.stroke_width)

        return to_tensor(image)
