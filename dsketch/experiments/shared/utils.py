import argparse
import importlib
import inspect
import json
import sys

import torchbearer


def list_class_names(clz, package):
    """
    Find sub-classes in a specific module and return their names

    Args:
        clz: the target superclass
        package: the module

    Returns: list of potential classes

    """

    def isclz(obj):
        if inspect.isclass(obj):
            return issubclass(obj, clz) and not obj == clz
        return False

    module = importlib.import_module(package)

    return [name for name, _ in inspect.getmembers(module, isclz)]


# this is to use with torchbearer imaging instead of the built-in to_file as it allows the state to be
# accessed in the name string. PR to torchbearer has been made to fix the original so this should be temporary.
def img_to_file(filename):
    from PIL import Image

    def handler(image, index, model_state):
        state = {}
        state.update(model_state)
        state.update(model_state[torchbearer.METRICS])

        string_state = {str(key): state[key] for key in state.keys()}

        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(filename.format(index=str(index), **string_state))

    return handler


# get a subparser for a specific named option
def get_subparser(name, parser):
    if name is None:
        return None

    subparsers = {
        name: subparser
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for name, subparser in action.choices.items()
    }
    subparser = subparsers[name]
    return subparser


def save_args(directory, name='cmd.txt'):
    with open(str(directory) + '/' + name, "w") as f:
        f.write(' '.join(sys.argv))


def save_model_info(model, directory, name='model-info.txt'):
    with open(str(directory) + '/' + name, "w") as f:
        f.write(str(model))


# argparse that doesn't show errors and exit
class FakeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        pass


ORIGINAL_Y_TRUE = torchbearer.state_key('original_y_true')


# Loader that transforms the target into the input, but maintains the labels in a separate key
def autoenc_loader(state):
    image, label = torchbearer.deep_to(next(state[torchbearer.ITERATOR]), state[torchbearer.DEVICE],
                                       state[torchbearer.DATA_TYPE])
    state[torchbearer.X] = image
    state[torchbearer.Y_TRUE] = image
    state[ORIGINAL_Y_TRUE] = label


def _parse_schedule(sched):
    if '@' in sched:
        factor, schtype = sched.split('@')
        factor = float(factor)
    else:
        factor, schtype = None, sched

    step_on_batch = False
    if schtype.endswith('B') or schtype.endswith('b'):
        step_on_batch = True
        schtype = schtype[:-1]

    if schtype.startswith('every'):
        step = int(schtype[5:])
        return torchbearer.callbacks.StepLR(step, gamma=factor, step_on_batch=step_on_batch)
    if schtype.startswith('inv'):
        gamma, power = (float(i) for i in schtype[3:].split(","))
        return torchbearer.callbacks.LambdaLR(lambda i: (1 + gamma * i) ** (- power), step_on_batch=step_on_batch)
    elif schtype.startswith('['):
        milestones = json.loads(schtype)
        return torchbearer.callbacks.MultiStepLR(milestones, gamma=factor, step_on_batch=step_on_batch)
    elif schtype == 'plateau':
        return torchbearer.callbacks.ReduceLROnPlateau(factor=factor, step_on_batch=step_on_batch)

    assert False


def parse_learning_rate_arg(learning_rate: str):
    """
    Parse a learning rate argument into an initial rate and an optional scheduler callback.

    Examples:
        0.1                        --  fixed lr
        0.1*0.2@every10            --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@every10B           --  decrease by factor=0.2 every 10 epochs
        0.1*0.2@[10,30,80]         --  decrease by factor=0.2 at epochs 10, 30, and 80
        0.1*0.2@[10,30,80]B        --  decrease by factor=0.2 at epochs 10, 30, and 80
        0.1*0.2@plateau            --  decrease by factor=0.2 on validation plateau
        0.1*inv0.0001,0.75B        --  decrease by using the old caffe inv rule each batch

    Args:
        learning_rate: lr string

    Returns:
        tuple of init_lr, [callback]
    """
    sp = str(learning_rate).split('*')
    initial = float(sp[0])

    if len(sp) == 1:
        return initial, []
    elif len(sp) == 2:
        return initial, [_parse_schedule(sp[1])]
    assert False

