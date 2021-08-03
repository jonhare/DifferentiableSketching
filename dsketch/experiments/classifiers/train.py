import argparse
import pathlib

import torch
import torch.optim as optim
import torchbearer as tb
from torchbearer.callbacks.checkpointers import Interval, MostRecent
from torchbearer.callbacks.csv_logger import CSVLogger

from dsketch.experiments.classifiers.models import get_model, model_choices
from dsketch.experiments.shared.args_datasets import get_dataset, dataset_choices, build_dataloaders
from dsketch.experiments.shared.utils import FakeArgumentParser, save_args, parse_learning_rate_arg


def add_shared_args(parser):
    parser.add_argument("--device", help='device to use', required=False, type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--model", type=str, help="name of model", choices=model_choices(), required=True)
    parser.add_argument("--dataset", type=str, help="name of model", choices=dataset_choices(), required=True)
    parser.add_argument("--epochs", default=10, help="number of epochs.", type=int)
    parser.add_argument("--snapshot_epochs", default=10, help="number of epochs to save snapshot.", type=int)
    parser.add_argument("--output", help="output location for logs and snapshots", required=True, type=pathlib.Path)
    parser.add_argument("--learning-rate", type=str, required=False, help="learning rate spec", default=0.001)
    parser.add_argument("--weight-decay", type=float, required=False, help="weight decay", default=0)
    parser.add_argument("--momentum", type=float, required=False, help="momentum (if SGD is used)", default=0.9)
    parser.add_argument("--optimiser", type=str, required=False, help="optimiser", default='Adam',
                        choices=['Adam', 'SGD'])


def add_sub_args(args, parser):
    if 'dataset' in args and args.dataset is not None:
        get_dataset(args.dataset).add_args(parser)

        
def build_distributed_dataloaders(args):
    ds = get_dataset(args.dataset)
    args.size = ds.get_size(args)

    train, valid, test = ds.create(args)

    sampler_train = torch.utils.data.distributed.DistributedSampler(train, shuffle=True)
    sampler_val = torch.utils.data.distributed.DistributedSampler(valid, shuffle=False)
    
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler_train)
    valloader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler_val)

    return trainloader, valloader, valloader
        
def main():
    fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
    add_shared_args(fake_parser)
    fake_args, _ = fake_parser.parse_known_args()

    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_sub_args(fake_args, parser)
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.output.mkdir(exist_ok=True, parents=True)
    save_args(args.output)
    
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

    
def main_worker(gpu, args):
    args.rank += gpu
    
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    args.device=cuda(gpu)
    
#     trainloader, valloader, testloader = build_dataloaders(args)

    path = str(args.output)

    model = torch.nn.parallel.DistributedDataParallel(get_model(args.model)(), device_ids=[gpu])

    init_lr, sched = parse_learning_rate_arg(args.learning_rate)

    if args.optimiser == 'Adam':
        opt = optim.Adam(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=init_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    
    
    ds = get_dataset(args.dataset)
    args.size = ds.get_size(args)

    train, valid, test = ds.create(args)

    sampler_train = torch.utils.data.distributed.DistributedSampler(train, shuffle=True)
    sampler_val = torch.utils.data.distributed.DistributedSampler(valid, shuffle=False)
    
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    
    trainloader = torch.utils.data.DataLoader(train, batch_size=per_device_batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler_train)
    valloader = torch.utils.data.DataLoader(valid, batch_size=per_device_batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler_val)
        
    @torchbearer.callbacks.on_start_epoch
    def sampler_set_epoch(state):
        sampler_train.set_epoch(state[torchbearer.EPOCH])
        sampler_val.set_epoch(state[torchbearer.EPOCH])

    callbacks = [
        sampler_set_epoch,
        Interval(filepath=path + '/model.{epoch:02d}.pt', period=10),
        MostRecent(filepath=path + '/model_final.pt'),
        CSVLogger(path + '/train-log.csv'),
        *sched
    ]

    trial = tb.Trial(model, opt, torch.nn.CrossEntropyLoss(), metrics=['loss', 'acc', 'lr'],
                     callbacks=callbacks).to(args.device)
    trial.with_generators(train_generator=trainloader, val_generator=valloader)
    trial.run(epochs=args.epochs, verbose=2)

        
# def main():
#     fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
#     add_shared_args(fake_parser)
#     fake_args, _ = fake_parser.parse_known_args()

#     parser = argparse.ArgumentParser()
#     add_shared_args(parser)
#     add_sub_args(fake_args, parser)
#     args = parser.parse_args()

#     trainloader, valloader, testloader = build_dataloaders(args)

#     args.output.mkdir(exist_ok=True, parents=True)
#     path = str(args.output)
#     save_args(args.output)

#     model = get_model(args.model)()

#     init_lr, sched = parse_learning_rate_arg(args.learning_rate)

#     if args.optimiser == 'Adam':
#         opt = optim.Adam(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
#     else:
#         opt = optim.SGD(model.parameters(), lr=init_lr, weight_decay=args.weight_decay, momentum=args.momentum)

#     callbacks = [
#         Interval(filepath=path + '/model.{epoch:02d}.pt', period=10),
#         MostRecent(filepath=path + '/model_final.pt'),
#         CSVLogger(path + '/train-log.csv'),
#         *sched
#     ]

#     trial = tb.Trial(model, opt, torch.nn.CrossEntropyLoss(), metrics=['loss', 'acc', 'lr'],
#                      callbacks=callbacks).to(args.device)
#     trial.with_generators(train_generator=trainloader, val_generator=valloader)
#     trial.run(epochs=args.epochs, verbose=2)

#     trial = tb.Trial(model, criterion=torch.nn.CrossEntropyLoss(), metrics=['loss', 'acc'],
#                      callbacks=[CSVLogger(path + '/test-log.csv')]).to(args.device)
#     trial.with_generators(test_generator=testloader)
#     trial.predict(verbose=2)
