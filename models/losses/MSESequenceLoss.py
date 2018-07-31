
import torch.nn as nn
import torch


class MSESequenceLoss(nn.Module):
    def __init__(self):
        super(MSESequenceLoss, self).__init__()

    def forward(self, inputs, targets):
        T = inputs.shape[1]
        if targets.shape[1] != T:
            f_0 = torch.unsqueeze(targets[:, 0, :, :, :], 1)
            targets = torch.cat([f_0, targets], dim=1)
        return torch.mean(inputs.sub(targets) ** 2)
