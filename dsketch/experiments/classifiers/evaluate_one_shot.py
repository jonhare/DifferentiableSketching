"""
Code for evaluating on the Omniglot 1-shot task exactly as described by Lake et al. Based on code from the official
repository: https://github.com/brendenlake/omniglot/tree/master/python/one-shot-classification
"""

# import numpy as np
import abc
import argparse
import copy
import pathlib
import sys
from os.path import join

import torch
import torchbearer
import torchvision.transforms.functional as F
from PIL import Image
from torch.nn import Module
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from dsketch.experiments.classifiers.models import OmniglotCNN, LakeThesisCNN
from dsketch.experiments.shared.args_datasets import C28pxOmniglotDataset, OmniglotDataset
from dsketch.experiments.shared.utils import FakeArgumentParser, list_class_names
from dsketch.utils.mod_haussdorff import binary_image_to_points, mod_hausdorff_distance
from ..characters import autoencoder

# Parameters
NRUN = 20  # number of classification runs
FNAME_LABEL = 'class_labels.txt'  # where class labels are stored for each run
FOLDER = 'omniglot-one-shot'

download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python/one-shot-classification/'
zips_md5 = {
    'all_runs': 'e8996daecdf12afeeb4a53a179f06b19'
}


def _check_integrity(root) -> bool:
    zip_filename = 'all_runs'
    if not check_integrity(join(root, zip_filename + '.zip'), zips_md5[zip_filename]):
        return False
    return True


def download(root) -> None:
    if _check_integrity(root):
        print('Files already downloaded and verified')
        return

    filename = 'all_runs'
    zip_filename = filename + '.zip'
    url = download_url_prefix + '/' + zip_filename
    download_and_extract_archive(url, root, filename=zip_filename, md5=zips_md5[filename])


def classification_run(base, folder, f_load, f_cost, ftype='cost'):
    # Compute error rate for one run of one-shot classification
    #
    # Input
    #  folder : contains images for a run of one-shot classification
    #  f_load : itemA = f_load('file.png') should read in the image file and process it
    #  f_cost : f_cost(itemA, itemB) should compute similarity between two images, using output of f_load
    #  ftype  : 'cost' if small values from f_cost mean more similar, or 'score' if large values are more similar
    #
    # Output
    #  perror : percent errors (0 to 100% error)
    #
    assert ((ftype == 'cost') | (ftype == 'score'))

    # get file names
    with open(base + '/' + folder + '/' + FNAME_LABEL) as f:
        content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    answers_files = copy.copy(train_files)
    test_files.sort()
    train_files.sort()
    ntrain = len(train_files)
    ntest = len(test_files)

    # load the images (and, if needed, extract features)
    train_items = [f_load(base + '/' + f) for f in train_files]
    test_items = [f_load(base + '/' + f) for f in test_files]

    # compute cost matrix
    cost_mat = torch.zeros((ntest, ntrain))
    for i in range(ntest):
        for c in range(ntrain):
            cost_mat[i, c] = f_cost(test_items[i], train_items[c])
    if ftype == 'cost':
        y_hat = torch.argmin(cost_mat, dim=1)
    elif ftype == 'score':
        y_hat = torch.argmax(cost_mat, dim=1)
    else:
        assert False

    # compute the error rate
    correct = 0.0
    for i in range(ntest):
        if train_files[y_hat[i]] == answers_files[i]:
            correct += 1.0
    pcorrect = 100 * correct / ntest
    perror = 100 - pcorrect

    return perror


class _Model(abc.ABC):
    def __init__(self, args):
        super().__init__()

    @abc.abstractmethod
    def load(self, fn):
        pass

    @abc.abstractmethod
    def cost(self):
        pass

    @classmethod
    def add_args(cls, parser):
        pass


class Hausdorff(_Model):
    def load(self, fn):
        img = Image.open(fn, mode='r').convert('L')
        img = F.to_tensor(img)
        return binary_image_to_points(img, invert=True)

    def cost(self):
        return mod_hausdorff_distance, 'cost'


class _Wrapper(Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl

    def forward(self, x):
        return self.mdl.get_feature(x)


class Network(_Model):
    def __init__(self, args):
        super().__init__(args)

        if (args.weights.parent / 'cmd.txt').exists():
            traincmd = (args.weights.parent / 'cmd.txt').read_text()

            if "OmniglotCNN" in traincmd:
                self.model = OmniglotCNN()
            elif "LakeThesisCNN" in traincmd:
                self.model = LakeThesisCNN()
            else:
                raise Exception("Unknown model type")

            state = torch.load(args.weights, map_location='cpu')
            self.model.load_state_dict(state[torchbearer.MODEL])
            self.model = _Wrapper(self.model)
        elif (args.weights.parent / 'train_cmd.txt').exists():
            traincmd = (args.weights.parent / 'train_cmd.txt').read_text().split(' ')[1:]
            aeargs = autoencoder.parse_args(traincmd)
            aeargs.device = "cpu"
            aeargs.weights = args.weights
            aeargs.size = args.size
            self.model = _Wrapper(autoencoder.build_model(aeargs))
        else:
            raise Exception("Could not find cmd.txt or train_cmd.txt file with model config")

        if args.size == 28:
            self.transform = C28pxOmniglotDataset.get_transforms(args)
        else:
            self.transform = OmniglotDataset.get_transforms(args)

        self.model.eval()

        self.measure = args.measure

    def load(self, fn):
        img = Image.open(fn, mode='r').convert('L')
        img = self.transform(img)
        img = img.unsqueeze(0)
        return self.model(img)[0].view(-1)
        # return self.model2.enc1[0:2](self.model(img).view(1, -1)).view(-1)

    def cost(self):
        if self.measure == 'cosine':
            return lambda a, b: torch.cosine_similarity(a, b, dim=0), 'score'
        if self.measure == 'euclidean':
            return torch.dist, 'cost'

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--weights", type=pathlib.Path, help="saved weights", required=True)
        parser.add_argument("--size", type=int, help="input image size", required=False, default=28, choices=[28, 105])
        parser.add_argument("--measure", type=str, help="similarity/distance", required=False, default='cosine',
                            choices=['cosine', 'euclidean'])


def run_eval(data_dir, f_load, f_cost, ftype):
    data_dir = join(data_dir, FOLDER)
    download(data_dir)

    if not _check_integrity(data_dir):
        raise RuntimeError('Dataset not found or corrupted.')

    perror = torch.zeros(NRUN)
    for r in range(1, NRUN + 1):
        rs = str(r)
        if len(rs) == 1:
            rs = '0' + rs
        perror[r - 1] = classification_run(str(data_dir), 'run' + rs, f_load, f_cost, ftype)
        print(" run " + str(r) + " (error " + str(perror[r - 1].item()) + "%)")
    total = torch.mean(perror)
    print(" average error " + str(total.item()) + "%")


def get_model(args):
    ds = getattr(sys.modules[__name__], args.model)
    return ds


def add_shared_args(parser):
    parser.add_argument("--model", type=str, help="name of model",
                        choices=list_class_names(_Model, __name__), required=True)
    parser.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                        default=pathlib.Path("./data/"), required=False)


def add_sub_args(args, parser):
    if 'model' in args and args.model is not None:
        get_model(args).add_args(parser)


def main():
    fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
    add_shared_args(fake_parser)
    fake_args, _ = fake_parser.parse_known_args()

    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_sub_args(fake_args, parser)
    args = parser.parse_args()

    model = get_model(args)(args)

    run_eval(args.dataset_root, model.load, *model.cost())
