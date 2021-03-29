import torch.nn as nn

from dsketch.experiments.characters.models.model_bases import Encoder
from dsketch.models.sketch_net import AgentCNN


class SimpleMLPEncoder(Encoder):
    def __init__(self, sz=28, hidden=128, latent=64):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(sz ** 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU())

    @staticmethod
    def _add_args(p):
        p.add_argument("--encoder-hidden", help="encoder hidden size", type=int, default=64, required=False)

    @staticmethod
    def create(args):
        return SimpleMLPEncoder(sz=args.size, hidden=args.encoder_hidden, latent=args.latent_size)

    def forward(self, inp, state=None):
        inp = inp.view(inp.shape[0], -1)
        return self.enc(inp)

    
class Conv2DEncoder(Encoder):
    def __init__(self, channels, latent_size=64):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)           
        )
        
#         self.fc1 = nn.Linear(320, latent_size) #320 for normal size MNIST, 74420 for scaled-up MNIST
#         self.fc2 = nn.Linear(hidden1, latent_size)
#         self.relu = nn.ReLU()
        
        
    @staticmethod
    def _add_args(p):
#         p.add_argument("--convEncoder-hidden1", help="Conv2D encoder hidden1", type=int, default=16384,
#                        required=False)
        p.add_argument("--encoder-channels", help="encoder input channels", type=int, default=1,
                       required=False)
        
    @staticmethod
    def create(args):
        return Conv2DEncoder(channels=args.encoder_channels, latent_size=args.latent_size)
    
    def forward(self, inp, state=None):
        inp = self.enc(inp)        
        inp = inp.view(inp.shape[0], -1)
#         inp = self.relu(self.fc1(inp))

        return inp
    
                   
    
class StrokeNetEncoder(Encoder):
    def __init__(self, channels=3, latent_size=1024, batchnorm=False):
        super().__init__()

        self.enc = AgentCNN(channels, batchnorm)
        self.fc = nn.Linear(16384, latent_size)  # hardcoded to match the dimensionality from strokeNet paper
        self.relu = nn.ReLU()
        
    @staticmethod
    def _add_args(p):
        p.add_argument("--encoder-channels", help="encoder AgentCNN number of channels", type=int, default=1,
                       required=False)
        p.add_argument("--batchnorm", help="use batchnorm after convs", default=False, required=False,
                       action='store_true')

    @staticmethod
    def create(args):
        return StrokeNetEncoder(channels=args.encoder_channels, latent_size=args.latent_size, batchnorm=args.batchnorm)

    def forward(self, inp, state=None):
        inp = self.enc(inp)
        inp = self.relu(self.fc(inp))
        return inp


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SimpleCNNEncoder(Encoder):
    def __init__(self, latent=64):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(1, 120, (5, 5), padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(120, 300, (5, 5), padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            Flatten(),
            nn.Linear(4800, latent)
        )

    @staticmethod
    def _add_args(p):
        pass

    @staticmethod
    def create(args):
        return SimpleCNNEncoder(latent=args.latent_size)

    def forward(self, inp, state=None):
        return self.enc(inp)


class BetterCNNEncoder(SimpleCNNEncoder):
    def __init__(self, latent=64):
        super().__init__()

        self.enc = nn.Sequential(
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
            Flatten(),
            nn.Linear(8 * 8 * 64, latent)
        )

    @staticmethod
    def create(args):
        return BetterCNNEncoder(latent=args.latent_size)

