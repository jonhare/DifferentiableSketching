import torch.nn as nn


class AgentConvBlock(nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(AgentConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, ksize, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nout, nout, ksize, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = self.lrelu1(self.conv1(x))
        h = self.lrelu2(self.conv2(h))
        return self.pool(h)


class AgentConvBlockBN(nn.Module):
    def __init__(self, nin, nout, ksize=3):
        super(AgentConvBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, ksize, padding=1)
        self.bn1 = nn.BatchNorm2d(nout)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nout, nout, ksize, padding=1)
        self.bn2 = nn.BatchNorm2d(nout)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = self.lrelu1(self.bn1(self.conv1(x)))
        h = self.lrelu2(self.bn2(self.conv2(h)))
        return self.pool(h)


class AgentCNN(nn.Module):
    """
    The AgentCNN network from the SketchNet paper. By default takes a [C, 256, 256] image and projects into a 256 dim
    vector.

    Args:
        channels: number of channels in input
    """

    def __init__(self, channels, batchnorm):
        super(AgentCNN, self).__init__()
        if batchnorm:
            acb = AgentConvBlockBN
        else:
            acb = AgentConvBlock

        self.down1 = acb(channels, 16)
        self.down2 = acb(16, 32)
        self.down3 = acb(32, 64)
        self.down4 = acb(64, 128)
        self.down5 = acb(128, 256)

    @staticmethod
    def _output_size(input_size):
        return int((input_size / 32) ** 2 * 256)

    def forward(self, x):
        h = self.down1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)

        return h.view(h.shape[0], -1)
