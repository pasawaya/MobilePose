
import torch.nn as nn


class MSESequenceLoss(nn.Module):
    def __init__(self):
        super(MSESequenceLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, inputs, targets):
        T = inputs.shape[1]
        if targets.shape[1] != T:
            targets = targets.repeat(1, T, 1, 1, 1)

        losses = []
        for i in range(inputs.shape[0]):
            for t in range(T):
                losses.append(self.criterion(inputs[i, t], targets[i, t]))
        return sum(losses)
