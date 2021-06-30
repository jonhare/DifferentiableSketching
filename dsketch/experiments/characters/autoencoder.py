import argparse
import pathlib

import torch
import torchbearer
import torchbearer as tb
from torchbearer.callbacks import Interval, CSVLogger
from torchbearer.callbacks import imaging

from .models import Encoder, Decoder, AutoEncoder, get_model, model_choices, VariationalAutoEncoder, LOGVAR, MU
from ..classifiers.models import get_model as get_classifier_model
from ..classifiers.models import model_choices as classifier_model_choices
from ..shared.args_datasets import build_dataloaders, dataset_choices, get_dataset
from ..shared.args_losses import get_loss as get_loss, build_loss
from ..shared.args_losses import loss_choices
from ..shared.metrics import ChamferMetric, ModifiedHausdorffMetric, ClassificationMetric
from ..shared.utils import img_to_file, get_subparser, save_args, save_model_info, FakeArgumentParser, autoenc_loader, \
    parse_learning_rate_arg


def train(args):
    args.output.mkdir(exist_ok=True, parents=True)
    save_args(args.output, name='train_cmd.txt')

    traingen, valgen, testgen = build_dataloaders(args)

    trial, model = build_trial(args)
    save_model_info(model, args.output)

    trial.to(args.device)
    trial.with_generators(train_generator=traingen, val_generator=valgen, test_generator=testgen)

    trial.run(epochs=args.epochs, verbose=2)

    state = trial.state_dict()
    string_state = {str(key): state[key] for key in state.keys()}
    torch.save(string_state, str(args.output) + '/model_final.pt')


def evaluate(args):
    args.output.mkdir(exist_ok=True, parents=True)
    save_args(args.output, name='eval_cmd.txt')

    traingen, valgen, testgen = build_dataloaders(args)

    trial, model = build_test_trial(args)
    model.to(args.device)
    save_model_info(model, args.output)

    trial.to(args.device)
    trial.with_generators(test_generator=testgen)
    trial.predict(verbose=2)


def build_trial(args):
    model = build_model(args)
    loss = build_loss(args)

    init_lr, sched = parse_learning_rate_arg(args.learning_rate)
    optim = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)

    inv = get_dataset(args.dataset).inv_transform

    callbacks = [
        Interval(filepath=str(args.output) + '/model_{epoch:02d}.pt', period=args.snapshot_interval),
        CSVLogger(str(args.output) + '/log.csv'),
        imaging.FromState(tb.Y_PRED, transform=inv).on_val().cache(args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/val_reconstruction_samples_{epoch:02d}.png')),
        imaging.FromState(tb.Y_TRUE, transform=inv).on_val().cache(args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/val_samples.png')),
        *model.get_callbacks(args),
        *sched,
    ]

    if args.variational:
        @torchbearer.callbacks.add_to_loss
        def add_kld_loss_callback(state):
            kl = torch.mean(0.5 * torch.sum(torch.exp(state[LOGVAR]) + state[MU] ** 2 - 1. - state[LOGVAR], 1))
            return kl

        callbacks.append(add_kld_loss_callback)

    trial = tb.Trial(model, optimizer=optim, criterion=loss, metrics=['loss', 'mse', 'lr'], callbacks=callbacks)
    trial.with_loader(autoenc_loader)

    return trial, model


def build_test_trial(args):
    model = build_model(args).to(args.device)
    inv = get_dataset(args.dataset).inv_transform

    callbacks = [
        CSVLogger(str(args.output) + '/log.csv'),
        imaging.FromState(tb.Y_PRED, transform=inv).on_test().cache(args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/test_reconstruction_samples.png')),
        imaging.FromState(tb.Y_TRUE, transform=inv).on_test().cache(args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/test_samples.png'))
    ]

    metrics = ['mse'] #Additional metrics: ChamferMetric(), ModifiedHausdorffMetric()

    if args.classifier_weights:
        classifier = get_classifier_model(args.classifier_model)().to(args.device)
        state = torch.load(args.classifier_weights, map_location=args.device)
        classifier.load_state_dict(state[tb.MODEL])
        metrics.append(ClassificationMetric(classifier))

    trial = tb.Trial(model, metrics=metrics, callbacks=callbacks)
    trial.with_loader(autoenc_loader)

    return trial, model


def build_model(args):
    if args.variational:
        # increase latent size of enc
        args.latent_size *= 2
        enc = get_model(args.encoder).create(args)
        args.latent_size //= 2
        # put it back to normal afterwards
        dec = get_model(args.decoder).create(args)
        ae = VariationalAutoEncoder(enc, dec, args.latent_size)
    else:
        enc = get_model(args.encoder).create(args)
        dec = get_model(args.decoder).create(args)
        ae = AutoEncoder(enc, dec)

    if 'weights' in args:
        state = torch.load(args.weights, map_location=args.device)
        ae.load_state_dict(state[tb.MODEL])

    return ae


def add_shared_args(parser):
    parser.add_argument("--variational", help="use a VAE", required=False, default=False, action='store_true')
    parser.add_argument("--encoder", help="encoder class", required=True, choices=model_choices(Encoder))
    parser.add_argument("--decoder", help="decoder class", required=True, choices=model_choices(Decoder))
    parser.add_argument("--dataset", help="dataset", required=True, choices=dataset_choices())
    parser.add_argument("--num-reconstructions", type=int, required=False, help="number of reconstructions to save",
                        default=100)
    parser.add_argument("--output", help='folder to store outputs', required=True, type=pathlib.Path)
    parser.add_argument("--device", help='device to use', required=False, type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')


def add_subparsers(parser, add_help=True):
    subparsers = parser.add_subparsers(dest='mode')
    train_parser = subparsers.add_parser("train", add_help=add_help)
    add_shared_args(train_parser)
    train_parser.add_argument("--loss", choices=loss_choices(), required=True, help="loss function")
    train_parser.add_argument("--epochs", type=int, required=False, help="number of epochs", default=10)
    train_parser.add_argument("--learning-rate", type=str, required=False, help="learning rate spec", default=0.001)
    train_parser.add_argument("--weight-decay", type=float, required=False, help="weight decay", default=0)
    train_parser.add_argument("--snapshot-interval", type=int, required=False,
                              help="interval between saving model snapshots", default=10)
    train_parser.set_defaults(perform=train)
    eval_parser = subparsers.add_parser("eval", add_help=add_help)
    add_shared_args(eval_parser)
    eval_parser.add_argument("--weights", help="model weights", type=pathlib.Path, required=True)
    eval_parser.add_argument("--classifier-weights", help="classifier weights", type=pathlib.Path, required=False)
    eval_parser.add_argument("--classifier-model", help="classifier implementation", type=str, required=False,
                             choices=classifier_model_choices())
    eval_parser.set_defaults(perform=evaluate)


def add_sub_args(args, parser):
    subparser = get_subparser(args.mode, parser)

    if 'encoder' in args and args.encoder is not None:
        get_model(args.encoder).add_args(subparser)
    if 'decoder' in args and args.decoder is not None:
        get_model(args.decoder).add_args(subparser)
    if 'dataset' in args and args.dataset is not None:
        get_dataset(args.dataset).add_args(subparser)

    if args.mode == 'train' and 'loss' in args and args.loss is not None:
        get_loss(args.loss).add_args(subparser)


def parse_args(argslist=None):
    fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
    add_subparsers(fake_parser, add_help=False)
    fake_args, _ = fake_parser.parse_known_args(argslist)

    if fake_args.mode is None:
        print("Mode must be specifed {train,eval}.")
        return

    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_subparsers(parser)
    add_sub_args(fake_args, parser)

    args = parser.parse_args(argslist)
    args.channels = get_dataset(args.dataset).get_channels(args)

    return args


def main():
    args = parse_args()

    args.perform(args)
