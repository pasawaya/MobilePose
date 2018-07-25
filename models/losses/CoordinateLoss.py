
import torch.nn as nn
import dsntnn


class CoordinateLoss(nn.Module):
    def __init__(self):
        super(CoordinateLoss, self).__init__()

    def forward(self, coords, heatmaps, targets):
        euc_loss = dsntnn.euclidean_losses(coords, targets)
        reg_loss = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=1.0)
        loss = dsntnn.average_loss(euc_loss + reg_loss)
        return loss
