

import torch.nn as nn
import torch.nn.functional as F
from models.modules.ResidualBlock import ResidualBlock
from models.modules.ConvGRU import ConvGRU


class RecurrentHourglass(nn.Module):
    def __init__(self, depth, hidden_channels, out_channels, device):
        super(RecurrentHourglass, self).__init__()

        self.depth = depth
        channels = hidden_channels + out_channels
        layers = (self.depth - 1) * [nn.ModuleList([ResidualBlock(channels, channels),
                                                    ConvGRU(channels, channels, 3, 1, device),
                                                    ResidualBlock(channels, channels)])]
        top_layer = [nn.ModuleList([ResidualBlock(channels, channels),
                                    ConvGRU(channels, channels, 3, 1, device),
                                    ResidualBlock(channels, out_channels)])]
        self.layers = nn.ModuleList(layers + top_layer)

    def recursive_forward(self, layer, x):
        x = F.max_pool2d(x, 2)
        x = self.layers[layer - 1][0](x)
        upper = self.layers[layer - 1][1](x)[-1]

        if layer == 0:
            out = self.layers[0][2](upper)
        else:
            out = self.layers[layer - 1][2](upper + self.recursive_forward(layer - 1, upper))

        out = F.upsample(out, scale_factor=2)
        return out

    def forward(self, x, b_t_1):
        return self.recursive_forward(self.depth, x)
