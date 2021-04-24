"""
This implements a training script for training the feature extractor using the Barlow-Twins self-supervised model
(https://arxiv.org/pdf/2103.03230.pdf), and then training the classifier layer on top (optionally) finetuning
the backbone.
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchbearer as tb
from torchbearer.callbacks.checkpointers import Interval, MostRecent
from torchbearer.callbacks.csv_logger import CSVLogger
from torchvision import transforms

from dsketch.experiments.classifiers.models import get_model
from dsketch.experiments.shared.args_datasets import get_dataset
from dsketch.experiments.shared.utils import FakeArgumentParser, save_args, parse_learning_rate_arg
from .train import add_shared_args as default_add_shared_args
from .train import add_sub_args


def add_shared_args(parser):
    default_add_shared_args(parser)
    parser.add_argument("--barlow-epochs", default=10, help="number of epochs for barlow twins SSL.", type=int)
    parser.add_argument("--barlow-batch-size", default=1024, help="Barlow twins SSL batch size.", type=int)
    parser.add_argument("--finetune", default=False, help="Should the bbackbone be finetuned after barlow SSL",
                        action='store_true')


class BarlowTwinsModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, y_a, y_b):
        # input is two randomly augmented versions of x

        # compute representations
        z_a = self.model.get_feature(y_a)  # NxD
        z_b = self.model.get_feature(y_b)  # NxD

        return z_a, z_b


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss:
    def __init__(self, lamda=5e-3):
        self.lamda = lamda

    def __call__(self, z_a, z_b, _):
        n, d = z_a.shape

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / n  # DxD
        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss


def train(args, model, model_loss, trainloader, valloader, epochs, name='model'):
    init_lr, sched = parse_learning_rate_arg(args.learning_rate)
    path = str(args.output)

    opt = build_optimiser(args, model, init_lr)
    callbacks = [
        Interval(filepath=path + '/' + name + '.{epoch:02d}.pt', period=10),
        MostRecent(filepath=path + '/' + name + '_final.pt'),
        CSVLogger(path + '/' + name + '-train-log.csv'),
        *sched
    ]

    metrics = ['loss', 'lr']
    if isinstance(model_loss, nn.CrossEntropyLoss):
        metrics.append('acc')

    trial = tb.Trial(model, opt, model_loss, metrics=metrics,
                     callbacks=callbacks).to(args.device)
    trial.with_generators(train_generator=trainloader, val_generator=valloader)
    trial.run(epochs=epochs, verbose=2)
    return trial


def build_optimiser(args, model, init_lr):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimiser == 'Adam':
        opt = optim.Adam(params, lr=init_lr, weight_decay=args.weight_decay)
    else:
        opt = optim.SGD(params, lr=init_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    return opt


def main():
    fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
    add_shared_args(fake_parser)
    fake_args, _ = fake_parser.parse_known_args()

    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_sub_args(fake_args, parser)
    args = parser.parse_args()

    rndtf = transforms.Compose([
        transforms.RandomAffine(10.0, translate=(0.1, 0.1), scale=(0.95, 1.01), shear=1),
        transforms.Lambda(lambda x: x * (1 + (torch.rand_like(x) - 0.5) / 10))
    ])
    args.additional_transforms = transforms.Lambda(lambda x: (rndtf(x), rndtf(x)))

    orig_batch_size = args.batch_size
    args.batch_size = args.barlow_batch_size
    trainloader, valloader, testloader = get_dataset(args.dataset).create(args)
    args.batch_size = orig_batch_size

    args.output.mkdir(exist_ok=True, parents=True)
    save_args(args.output)

    model = get_model(args.model)()
    btmodel = BarlowTwinsModel(model)

    model_loss = nn.CrossEntropyLoss()
    btmodel_loss = BarlowTwinsLoss()

    train(args, btmodel, btmodel_loss, trainloader, valloader, args.barlow_epochs, name='btmodel')

    model.lock_features(not args.finetune)

    trainloader, valloader, testloader = get_dataset(args.dataset).create(args)  # reload data with other batch size
    train(args, model, model_loss, trainloader, valloader, args.epochs, name='model')

    trial = tb.Trial(model, criterion=torch.nn.CrossEntropyLoss(), metrics=['loss', 'acc'],
                     callbacks=[CSVLogger(str(args.output) + '/test-log.csv')]).to(args.device)
    trial.with_generators(test_generator=testloader)
    trial.predict(verbose=2)
