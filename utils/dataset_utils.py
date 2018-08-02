
import numpy as np
from skimage.feature import plot_matches
from skimage.draw import circle, line
from skimage.io import imshow
from matplotlib import pyplot as plt
from .evaluation import *
import torch


def compute_mean(dataset):
    mean, std = np.zeros(3, dtype=np.float), np.zeros(3, dtype=np.float)
    for i in range(len(dataset)):
        video, _, _, _ = dataset[i]
        for t in range(len(video)):
            im = video[t]
            im = np.moveaxis(im, 2, 0)
            im = im.reshape(im.shape[0], -1)
            mean += np.mean(im, axis=1)
            std += np.std(im, axis=1)
    mean = (mean / 255.) / float(len(dataset))
    std = (std / 255.) / float(len(dataset))
    return mean, std


def draw_skeleton(image, coordinates):
    limbs = [(0, 1), (1, 2), (2, 3), (2, 8), (3, 8), (3, 4), (4, 5), (8, 9), (8, 12),
             (8, 13), (10, 11), (11, 12), (12, 13), (6, 7), (6, 13)]

    coordinates = coordinates.astype(np.int)
    for i, j in limbs:
        rr, cc = line(coordinates[i, 0], coordinates[i, 1], coordinates[j, 0], coordinates[j, 1])
        rr, cc = np.clip(rr, 0, image.shape[1] - 1), np.clip(cc, 0, image.shape[0] - 1)
        image[rr, cc] = (0, 255, 0)
    return image


def visualize(video, labels, outputs):
    if video.shape[1] != outputs.shape[1]:
        f_0 = torch.unsqueeze(video[:, 0, :, :, :], 1)
        video = torch.cat([f_0, video], dim=1)

    if labels.shape[1] != outputs.shape[1]:
        l_0 = torch.unsqueeze(labels[:, 0, :, :, :], 1)
        labels = torch.cat([l_0, labels], dim=1)

    batch_size, n_stages, n_joints = labels.shape[0], labels.shape[1], labels.shape[2]

    video = video.detach().numpy().copy().astype(np.uint8)
    video = np.moveaxis(video, 2, 4)

    labels = labels.detach()
    outputs = outputs.detach()

    image_size = video.shape[-2]
    label_size = labels.shape[-2]
    r = image_size / label_size

    for i in range(batch_size):
        batch_gt_coords = get_preds(labels[i, :, :, :, :]).numpy()
        batch_pred_coords = get_preds(outputs[i, :, :, :, :]).numpy()

        for t in range(n_stages):
            frame = video[i, t, :, :, :]
            frame_gt_coords = np.flip(batch_gt_coords[t, :, :] * r, axis=1)
            frame_pred_coords = np.flip(batch_pred_coords[t, :, :] * r, axis=1)

            gt_frame = draw_skeleton(frame.copy(), frame_gt_coords)
            pred_frame = draw_skeleton(frame.copy(), frame_pred_coords)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            matches = np.array([np.linspace(0, n_joints - 1, n_joints, dtype=np.int),
                                np.linspace(0, n_joints - 1, n_joints, dtype=np.int)]).T
            plot_matches(ax, gt_frame, pred_frame, frame_gt_coords, frame_pred_coords, np.array([]),
                         keypoints_color='red')
            plt.show()


def compute_label_map(x, y, size, sigma, stride, offset, background):
    if len(x.shape) < 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    t = x.shape[0]
    n_joints = x.shape[1]
    label_size = np.floor(size / stride).astype(int) + offset
    label_map_joints = n_joints + 1 if background else n_joints
    label_map = np.zeros((t, label_map_joints, label_size, label_size))
    start = (stride / 2.) - 0.5
    for t in range(t):
        for p in range(n_joints):
            center_x, center_y = x[t, p], y[t, p]
            X, Y = np.meshgrid(np.linspace(0, label_size, label_size), np.linspace(0, label_size, label_size))
            X = (X - 1) * stride + start - center_x
            Y = (Y - 1) * stride + start - center_y
            d2 = X * X + Y * Y
            exp = d2 * 0.5 / sigma / sigma
            label = np.exp(-exp)
            label[label < 0.01] = 0
            label[label > 1] = 1
            label_map[t, p, :, :] = label
    return torch.from_numpy(label_map).float()


def compute_center_map(size, sigma=21):
    shape = (size, size)
    x, y = size / 2, size / 2
    X, Y = np.meshgrid(np.linspace(0, shape[0], shape[0]), np.linspace(0, shape[1], shape[1]))
    X = X - x
    Y = Y - y
    d2 = X * X + Y * Y
    exp = d2 * 0.5 / sigma / sigma
    center_map = np.exp(-exp)
    center_map[center_map < 0.01] = 0
    center_map[center_map > 1] = 1
    center_map = np.expand_dims(center_map, axis=0)
    return torch.from_numpy(center_map).float()
