
import torch.nn as nn
import dsntnn
import torch


class CoordinateLoss(nn.Module):
    def __init__(self):
        super(CoordinateLoss, self).__init__()

    def forward(self, heatmaps, coords, targets):
        batch_size = coords.shape[0]
        n_stages = coords.shape[1]

        if len(targets.shape) != len(coords.shape):
            targets = torch.unsqueeze(targets, 1)
            targets = targets.repeat(1, n_stages, 1, 1)

        losses = []
        for i in range(batch_size):
            euc_loss = dsntnn.euclidean_losses(coords[i, :, :, :], targets[i, :, :, :])
            reg_loss = dsntnn.js_reg_losses(heatmaps[i, :, :, :, :], targets[i, :, :, :], sigma_t=1.0)
            losses.append(dsntnn.average_loss(euc_loss + reg_loss))
        return sum(losses) / batch_size
