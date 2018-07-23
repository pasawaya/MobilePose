
import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import initialize_weights_kaiming


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        hidden_channels = int(out_channels / 2.)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1)

        self.apply(initialize_weights_kaiming)

    def forward(self, x):
        residual = x
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out
