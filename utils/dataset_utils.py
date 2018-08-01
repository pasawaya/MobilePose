
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


def visualize_skeleton(image, x, y, vis):
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


def visualize_map(video, labels):
    batch_size = labels.shape[0]
    n_stages = labels.shape[1]
    n_joints = labels.shape[2]

    video = video.detach().numpy().copy().astype(np.uint8)
    video = np.moveaxis(video, 2, 4)
    labels = labels.detach()

    image_size = video.shape[-2]
    label_size = labels.shape[-2]
    r = image_size / label_size

    for i in range(batch_size):
        batch_coords = get_preds(labels[i, :, :, :, :]).numpy()
        for t in range(n_stages):
            frame = video[i, t, :, :, :]
            frame_label = labels[i, t, :, :, :].numpy() * 255.
            frame_coords = batch_coords[t, :, :] * r

            for p in range(n_joints):
                joint_label = frame_label[p, :, :]
                joint_label = np.expand_dims(joint_label, axis=2)
                joint_label = np.clip(joint_label, 0, 255).astype(np.uint8)
                joint_label = cv2.cvtColor(joint_label, cv2.COLOR_GRAY2RGB)
                joint_label = cv2.resize(joint_label, (image_size, image_size))

                joint_frame = cv2.addWeighted(frame, 0.5, joint_label, 0.5, 0)

                joint_coords = frame_coords[p, :]
                rr, cc = circle(joint_coords[1], joint_coords[0], 4)
                joint_frame[rr, cc] = (255, 0, 0)

                imshow(joint_frame)
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
