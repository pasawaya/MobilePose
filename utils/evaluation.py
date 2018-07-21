
import torch


def accuracy(inputs, targets, r=0.2):
    batch_size = inputs.shape[0]
    n_stages = inputs.shape[1]
    n_joints = inputs.shape[2]

    inputs = inputs.detach()
    targets = targets.detach()

    n_correct = 0
    n_total = batch_size * n_stages * n_joints

    for i in range(batch_size):
        gt = to_coordinates(torch.squeeze(targets[i], 0))

        w = gt[:, 0].max() - gt[:, 0].min()
        h = gt[:, 1].max() - gt[:, 1].min()
        threshold = r * max(w, h)

        # For each stage in the example...
        for j in range(n_stages):
            predictions = to_coordinates(inputs[i, j])
            score = torch.norm(predictions.sub(gt), 2, dim=1)
            n_correct += (score <= threshold).sum()

    return float(n_correct) / float(n_total)


# Get x, y coordinates of joint predictions from heatmap
def to_coordinates(inputs):
    maxes, indices = torch.max(inputs.view(inputs.size(0), -1), 1)
    maxes = maxes.view(inputs.size(0), 1)
    indices = indices.view(inputs.size(0), 1)

    predictions = indices.repeat(1, 2).float()

    predictions[:, 0] = predictions[:, 0] % inputs.size(2)
    predictions[:, 1] = torch.floor((predictions[:, 1]) / inputs.size(2))

    pred_mask = maxes.gt(0).repeat(1, 2).float()
    predictions *= pred_mask
    return predictions