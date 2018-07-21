

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import initialize_weights_kaiming
from .modules.RecurrentHourglass import RecurrentHourglass
from .modules.ResidualBlock import ResidualBlock


class PretrainRecurrentStackedHourglass(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, T=10, depth=4):
        super(PretrainRecurrentStackedHourglass, self).__init__()

        self.T = T

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.res1 = ResidualBlock(64, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, hidden_channels)

        self.hg_1 = RecurrentHourglass(depth, hidden_channels, out_channels, device)
        self.hg_t = RecurrentHourglass(depth, hidden_channels, out_channels, device)

        self.apply(initialize_weights_kaiming)

    def forward(self, x, centers):
        centers = F.avg_pool2d(centers, 4)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.res1(x)
        x = F.max_pool2d(x, 2)
        x = self.res2(x)
        x = self.res3(x)

        b_1 = self.hg_1(x)
        beliefs = [b_1]

        b_t_1 = b_1
        for t in range(self.T - 1):
            x = torch.cat([x, b_t_1], dim=1)
            b_t = self.hg_t(x, b_t_1)
            beliefs.append(b_t)
            b_t_1 = b_t

        out = torch.stack(beliefs, 1)
        return out
