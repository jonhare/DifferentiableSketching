import sys

import lpips
import torch
import torch.nn.functional as F

from dsketch.experiments.shared.utils import list_class_names
from dsketch.losses import BlurredMSE, PyramidMSE, DoGPyramidMSE

from torchvision.transforms import transforms
import torchvision.models as models
import torch.nn as nn


class _Loss:
    def __init__(self, args):
        if 'variational' in args and args.variational is True:
            self.imagewise = True
        else:
            self.imagewise = False

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, input, target):
        pass


class MSELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, input, target):
        if self.imagewise:
            return F.mse_loss(input, target, reduction='sum') / input.shape[0]
        return F.mse_loss(input, target)


class BCELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, input, target):
        if self.imagewise:
            return F.binary_cross_entropy(input, target, reduction='sum') / input.shape[0]
        return F.binary_cross_entropy(input, target)


class LPIPSLoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

        self.loss = lpips.LPIPS(net=args.net).to(args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--net", help="network for pips [alex or vgg]", type=str, default='vgg', required=False)

    def __call__(self, input, target):
        input2 = (input - 0.5) * 2
        target2 = (target - 0.5) * 2

        if input2.ndim == 3:
            input2 = input2.unsqueeze(0)
            target2 = target2.unsqueeze(0)

        if input2.shape[1] != 3:
            input2 = torch.cat([input2] * 3, dim=1)
            target2 = torch.cat([target2] * 3, dim=1)

        return self.loss(input2, target2)



VGG16_LAYER_KEYS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
VGG16_LAYERS = {'relu1_2': (4, 64, 1),
                'relu2_2': (9, 128, 2),
                'relu3_3': (16, 256, 4),
                'relu4_3': (23, 512, 8),
                'relu5_3': (30, 512, 16)}

def createVGG16FE(nettype, dev='cuda:0'):
    if nettype == 'sin':
        print("sin weights enabled")
        # download model from URL manually and save to desired location
        filepath = "/ssd//vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth"

        assert os.path.exists(
            filepath), "Please download the VGG model yourself from the following link and save it locally: " \
                       "https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be " \
                       "downloaded automatically like the other models)"

        vgg16 = models.vgg16(pretrained=False)
        checkpoint = torch.load(filepath, map_location=dev)
        state_dict = checkpoint['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace(".module", '') # removing ‘.moldule’ from key] # remove module.
            new_state_dict[name] = v
        
        vgg16.load_state_dict(new_state_dict)
    else:
        print("still imnet")
        vgg16 = models.vgg16(pretrained=True)


    class Wrapped(nn.Module):
        def __init__(self, net):
            super().__init__()

            layer = 'relu5_3'
            layidx, fms, ds = VGG16_LAYERS[layer]

            self.net = net.features[:layidx]
            
            self.fmaps = {}
            for i in range(len(VGG16_LAYER_KEYS)):
                key = VGG16_LAYER_KEYS[i]
                lay = VGG16_LAYERS[key][0]
                self.net[lay - 1].register_forward_hook(self.make_hook(key))

            for param in net.parameters():
                param.requires_grad = False

        def make_hook(self, name):
            def hook(module, input, output):
                self.fmaps[name] = output
            return hook

        def forward(self, x):
            self.fmaps = {}
            return self.net(x)

    return Wrapped(vgg16).to(dev)


def spatial_average(in_tens, keepdim=True):
    if len(in_tens.shape) == 2:
        in_tens = in_tens.view(in_tens.shape[0], 1, 1, 1)

    return in_tens.mean([2, 3], keepdim=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class FeatureMapLoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

        self.net = createVGG16FE(args.net, args.device)
        self.weights=[1,1,1,1,1]
        self.coef = 1

    @staticmethod
    def add_args(p):
        p.add_argument("--net", help="network weights", type=str, default='imnet', choices=['imnet', 'sin'], required=False)

    def __call__(self, input, target):
        if input.ndim == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)

        if input.shape[1] != 3:
            input = torch.cat([input] * 3, dim=1)
            target = torch.cat([target] * 3, dim=1)

        input = IMAGENET_NORM(input)
        target = IMAGENET_NORM(target)


        self.net(input)
        outs0 = self.net.fmaps
        
        self.net(target)
        outs1 = self.net.fmaps

        feats0, feats1, diffs = {}, {}, {}
        for kk in outs0.keys():
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in outs0.keys()]

        val = self.weights[0] * res[0]
        for i in range(1, min(len(res), len(self.weights))):
            val += self.weights[i] * res[i]

        return self.coef * val.mean()


class PyrMSELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)
        self.loss = PyramidMSE(args.channels, args.octaves, args.intervals, downsample=args.downsample,
                               symmetric=args.no_symmetric).to(args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--octaves", help="number of octaves", type=int, default=3, required=False)
        p.add_argument("--intervals", help="number of intervals", type=int, default=2, required=False)
        p.add_argument("--downsample", help="enable pyramid mode", action='store_true', required=False)
        p.add_argument("--no-symmetric", help="disable symmetric mode", action='store_false', required=False)

    def __call__(self, input, target):
        if input.ndim == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.loss(input, target)


class BlurredMSELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)
        self.loss = BlurredMSE(args.channels, args.blur_sigma, symmetric=args.no_symmetric).to(args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--blur-sigma", help="sigma for blurring in loss", type=float, default=1.0, required=False)
        p.add_argument("--no-symmetric", help="disable symmetric mode", action='store_false', required=False)

    def __call__(self, input, target):
        if input.ndim == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        return self.loss(input, target)


class DoGPyrMSELoss(PyrMSELoss):
    def __init__(self, args):
        args.no_symmetric = None
        super().__init__(args)
        self.loss = DoGPyramidMSE(args.channels, args.octaves, args.intervals, downsample=args.downsample).to(
            args.device)

    @staticmethod
    def add_args(p):
        p.add_argument("--octaves", help="number of octaves", type=int, default=3, required=False)
        p.add_argument("--intervals", help="number of intervals", type=int, default=2, required=False)
        p.add_argument("--downsample", help="enable pyramid mode", action='store_true', required=False)


def get_loss(name):
    los = getattr(sys.modules[__name__], name)
    if not issubclass(los, _Loss):
        raise TypeError()
    return los


def loss_choices():
    return list_class_names(_Loss, __name__)


def build_loss(args):
    return get_loss(args.loss)(args)
