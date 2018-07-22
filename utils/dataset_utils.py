
import numpy as np
from skimage.draw import circle, line
from skimage.io import imshow
from matplotlib import pyplot as plt
from .evaluation import *
import torch
import cv2


def compute_mean(dataset):
    mean, std = np.zeros(3, dtype=np.float), np.zeros(3, dtype=np.float)

    for i in range(len(dataset)):
        print(i)
        im, _, _, _ = dataset[i]
        im = np.moveaxis(im, 2, 0)
        im = im.reshape(im.shape[0], -1)
        mean += np.mean(im, axis=1)
        std += np.std(im, axis=1)
    mean = (mean / 255.) / float(len(dataset))
    std = (std / 255.) / float(len(dataset))
    return mean, std


def to_numpy(data):
    n = len(data)
    x, y, vis = np.zeros(n), np.zeros(n), np.zeros(n)
    for p in range(n):
        x[p] = data[str(p)][0]
        y[p] = data[str(p)][1]
        vis[p] = data[str(p)][2]
    return x, y, vis


def visualize_input(image, x, y, vis):
    limbs = [(0, 1), (1, 2), (2, 3), (2, 8), (3, 8), (3, 4), (4, 5), (8, 9), (8, 12),
             (8, 13), (10, 11), (11, 12), (12, 13), (6, 7), (6, 13)]

    image = image.astype(np.uint8)

    for p_x, p_y, p_vis in zip(x, y, vis):
        if p_vis:
            rr, cc = circle(p_y, p_x, 4)
            image[rr, cc] = (255, 255, 255)

    for i, j in limbs:
        if vis[i] and vis[j]:
            rr, cc = line(y[i], x[i], y[j], x[j])
            image[rr, cc] = (0, 255, 0)

    imshow(image)
    plt.show()


def visualize_label_map(image, label_map):
    label_map = label_map.numpy() * 255
    label_map = np.sum(label_map, axis=1, dtype=np.uint8, keepdims=True)
    label_map = np.repeat(np.moveaxis(label_map, 1, 3), 3, axis=3)
    label_map = np.squeeze(label_map, 0)
    label_map = cv2.resize(label_map, (image.shape[0], image.shape[1]))

    image = cv2.addWeighted(image, 0.5, label_map, 0.5, 0)
    imshow(image)
    plt.show()


def visualize_center_map(image, center_map):
    center_map = center_map.numpy() * 255.
    center_map = np.moveaxis(center_map, 0, 2).astype(np.uint8).copy()
    center_map = np.repeat(center_map, 3, axis=2)
    image = cv2.addWeighted(image, 0.5, center_map, 0.5, 0)
    imshow(image)
    plt.show()


def visualize_truth(video, labels):
    video = video.detach().numpy().copy().astype(np.uint8)
    video = np.moveaxis(video, 1, 3)

    labels = labels.detach()
    labels = torch.squeeze(labels, 1)
    coords = get_preds(labels) * (256. / 64.)

    size = video.shape[-2]
    for t in range(video.shape[0]):
        frame = video[t].copy()

        heatmap = torch.squeeze(labels[t], 0).numpy() * 255.
        heatmap = np.sum(heatmap, axis=0)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
        heatmap = cv2.resize(heatmap, (size, size))
        frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
        for (x, y) in coords[t]:
            cv2.circle(frame, (x, y), 2, (0, 255, 0))
        imshow(frame)
        plt.show()


def compute_label_map(x, y, visibility, size=256, sigma=7, stride=4):
    if len(x.shape) < 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        visibility = np.expand_dims(visibility, 0)

    t = x.shape[0]
    n_joints = x.shape[1]
    label_size = np.floor(size / stride).astype(int)
    label_map = np.zeros((t, n_joints + 1, label_size, label_size))
    start = (stride / 2.) - 0.5
    for t in range(t):
        for p in range(n_joints):
            if visibility[t, p] > 0:
                center_x, center_y = x[t, p], y[t, p]
                X, Y = np.meshgrid(np.linspace(0, label_size, label_size), np.linspace(0, label_size, label_size))
                X = (X - 1) * stride + start - center_x
                Y = (Y - 1) * stride + start - center_y
                d2 = X * X + Y * Y
                exp = d2 * 0.5 / sigma / sigma
                label = np.exp(-exp)
                label[label < 0.01] = 0
                label[label > 1] = 1
            else:
                label = np.zeros((label_size, label_size))
            label_map[t, p, :, :] = label
    return torch.from_numpy(label_map).float()


def compute_center_map(size=256, sigma=21):
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
