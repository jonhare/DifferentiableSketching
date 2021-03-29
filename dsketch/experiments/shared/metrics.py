import warnings

import torch
from torchbearer import Metric
import torchbearer
from torchbearer.metrics import CategoricalAccuracy, mean, running_mean

from dsketch.experiments.shared import utils
from dsketch.losses import chamfer
from dsketch.utils.mod_haussdorff import binary_image_to_points, mod_hausdorff_distance

HARDRASTER = torchbearer.state_key('hardraster')
SQ_DISTANCE_TRANSFORM = torchbearer.state_key('squared_distance_transform')
Y_PRED_CLASSES = torchbearer.state_key('y_pred_classes')


@running_mean
@mean
class ClassificationMetric(Metric):
    def __init__(self, classification_model):
        super().__init__("recon_class_acc")
        self.classification_model = classification_model

    def process(self, state):
        y_pred = self.classification_model(state[torchbearer.Y_PRED])  # take the reconstruction and classify it
        y_true = state[utils.ORIGINAL_Y_TRUE]

        if len(y_true.shape) == 2:
            _, y_true = torch.max(y_true, 1)

        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()


@running_mean
@mean
class ChamferMetric(Metric):
    def __init__(self):
        super().__init__('chamfer')
        self.warn = False

    def process(self, state):
        if not self.warn and not (state[torchbearer.Y_TRUE] == state[torchbearer.Y_TRUE].bool()).all():
            warnings.warn('Using chamfer distance on non-binary target images does not make sense')
            self.warn = True

        inp = state[SQ_DISTANCE_TRANSFORM]
        tgt = state[torchbearer.Y_TRUE]
        return chamfer(inp, tgt, dt2_fcn=None, ras_fcn=None, symmetric=False)


@running_mean
@mean
class ModifiedHausdorffMetric(Metric):
    def __init__(self):
        super().__init__('modified_hausdorff')
        self.warn = False

    def process(self, state):
        if not self.warn and not (state[torchbearer.Y_TRUE] == state[torchbearer.Y_TRUE].bool()).all():
            warnings.warn('Using modified Hausdorff distance on non-binary target images does not make sense')
            self.warn = True

        pred = binary_image_to_points(state[HARDRASTER])
        target = binary_image_to_points(state[torchbearer.Y_TRUE])

        return mod_hausdorff_distance(pred, target)
