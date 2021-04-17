import importlib

import torch.nn as nn
import torch.nn.functional as F

from dsketch.experiments.shared.utils import list_class_names


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 30, (5, 5), padding=0, stride=1)
        self.conv2 = nn.Conv2d(30, 15, (5, 5), padding=0, stride=1)
        self.fc1 = nn.Linear(6000, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)

    def lock_features(self, lock):
        self.conv1.requires_grad_(not lock)
        self.conv2.requires_grad_(not lock)
        self.fc1.requires_grad_(not lock)
        self.fc2.requires_grad_(not lock)

    def get_feature(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

    def forward(self, x):
        out = self.get_feature(x)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class ScaledMnistCNN(MnistCNN):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 30, (5, 5), padding=0, stride=2)
        self.conv2 = nn.Conv2d(30, 15, (5, 5), padding=0, stride=2)
        self.fc1 = nn.Linear(55815, 128)


class OmniglotCNN(nn.Module):
    """
    Omniglot DCN as described in the sup. mat. of the paper.

    This is the "larger" variant for the full 30 alphabet pretraining on 28x28 images. I've guessed there was no zero
    padding and the dropout probability was 0.5.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 120, (5, 5), padding=0, stride=1)
        self.conv2 = nn.Conv2d(120, 300, (5, 5), padding=0, stride=1)
        self.mp = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(30000, 3000)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(3000, 964)

    def lock_features(self, lock):
        self.conv1.requires_grad_(not lock)
        self.conv2.requires_grad_(not lock)
        self.fc1.requires_grad_(not lock)

    def get_feature(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.mp(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        return out

    def forward(self, x):
        out = self.get_feature(x)
        out = self.drop(out)
        out = self.fc2(out)
        return out


class LakeThesisCNN(nn.Module):
    """
    Omniglot CCN as described in Lake's thesis

    This is for the full 30 alphabet pretraining on 28x28 images. I've guessed there was no zero padding and the dropout
    probability was 0.5.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 200, (10, 10), padding=0, stride=1)
        self.mp = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(16200, 400)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(400, 964)

    def lock_features(self, lock):
        self.conv1.requires_grad_(not lock)
        self.fc1.requires_grad_(not lock)

    def get_feature(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.mp(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        return out

    def forward(self, x):
        out = self.get_feature(x)
        out = self.drop(out)
        out = self.fc2(out)
        return out


class _BetterCNN(nn.Module):
    def __init__(self, nclasses):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.fc2 = nn.Linear(1024, nclasses)

    def lock_features(self, lock):
        self.cnn.requires_grad_(not lock)
        self.fc1.requires_grad_(not lock)

    def get_feature(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.get_feature(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class MNISTBetterCNN(_BetterCNN):
    def __init__(self):
        super().__init__(10)


class OmniglotBetterCNN(_BetterCNN):
    def __init__(self):
        super().__init__(964)


def get_model(name):
    # load a model class by name
    module = importlib.import_module(__package__ + '.models')
    return getattr(module, name)


def model_choices():
    return list(filter(lambda x: not x.startswith('_'), list_class_names(nn.Module, __package__ + '.models')))
