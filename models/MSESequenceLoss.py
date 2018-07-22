
import torch.nn as nn
import torch


class MSESequenceLoss(nn.Module):
    def __init__(self):
        super(MSESequenceLoss, self).__init__()

    def forward(self, inputs, targets):
        T = inputs.shape[1]
        if targets.shape[1] != T:
            targets = targets.repeat(1, T, 1, 1, 1)

        return torch.mean(inputs.sub(targets) ** 2)
