
import torch.nn as nn


class MSESequenceLoss(nn.Module):
    def __init__(self):
        super(MSESequenceLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        T = inputs.shape[1]
        if targets.shape[1] != T:
            targets = targets.repeat(1, T, 1, 1, 1)

        losses = []
        for i in range(batch_size):
            for t in range(T):
                losses.append(self.criterion(inputs[i, t], targets[i, t]))
        return sum(losses) / (batch_size * T)
