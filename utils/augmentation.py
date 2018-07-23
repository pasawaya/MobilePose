
import numpy as np
from skimage.transform import resize, rotate
from random import random, uniform
import torch


class ImageTransformer(object):
    def __init__(self, output_size=256, r_pad=3.,
                 p_scale=1.0, p_flip=0.5, p_rotate=1.0,
                 min_scale=0.7, max_scale=1.3,
                 max_degree=20,
                 min_jitter=0.8, max_jitter=1.2,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):

        self.output_size = output_size
        self.mean = mean
        self.std = std

        self.p_scale = p_scale
        self.p_flip = p_flip
        self.p_rotate = p_rotate

        self.max_degree = max_degree
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter

        self.pad = self.output_size / r_pad

    def __call__(self, image, x, y, visibility, bbox=None):
        if bbox is None:
            x_min, x_max = max(0, np.amin(x) - self.pad), min(image.shape[1] - 1, np.amax(x) + self.pad)
            y_min, y_max = max(0, np.amin(y) - self.pad), min(image.shape[0] - 1, np.amax(y) + self.pad)
            bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)

        if random() < self.p_scale:
            f_xy = self.min_scale + (self.max_scale - self.min_scale) * random()
            image, bbox, x, y = self.scale(image, bbox, x, y, f_xy)

        if random() < self.p_flip:
            image, bbox, x, y = self.flip(image, bbox, x, y)

        image, x, y = self.crop(image, bbox, x, y, self.output_size)

        if random() < self.p_rotate:
            angle = -self.max_degree + 2 * self.max_degree * random()
            image, bbox, x, y = self.rotate(image, bbox, x, y, angle)

        image = self.color_jitter(image, self.min_jitter, self.max_jitter)
        image = self.normalize(image, self.mean, self.std)
        return self.to_torch(image), x, y, visibility

    @staticmethod
    def rotate(image, bbox, x, y, angle):
        o_x, o_y = (np.array(image.shape[:2][::-1]) - 1) / 2.
        image = rotate(image, angle, preserve_range=True).astype(np.uint8)
        r_x, r_y = (np.array(image.shape[:2][::-1]) - 1) / 2.

        angle = - np.deg2rad(angle)

        x = r_x + np.cos(angle) * (x - o_x) - np.sin(angle) * (y - o_y)
        y = r_y + np.sin(angle) * (x - o_x) + np.cos(angle) * (y - o_y)

        bbox[0] = r_x + np.cos(angle) * (bbox[0] - o_x) + np.sin(angle) * (bbox[1] - o_y)
        bbox[1] = r_y + -np.sin(angle) * (bbox[0] - o_x) + np.cos(angle) * (bbox[1] - o_y)
        bbox[2] = r_x + np.cos(angle) * (bbox[2] - o_x) + np.sin(angle) * (bbox[3] - o_y)
        bbox[3] = r_y + -np.sin(angle) * (bbox[2] - o_x) + np.cos(angle) * (bbox[3] - o_y)
        return image, bbox, x.astype(np.int), y.astype(np.int)

    @staticmethod
    def crop(image, bbox, x, y, length):
        x, y, bbox = x.astype(np.int), y.astype(np.int), bbox.astype(np.int)

        x_min, y_min, x_max, y_max = bbox
        w, h = x_max - x_min, y_max - y_min

        # Crop image to bbox
        image = image[y_min:y_min + h, x_min:x_min + w, :]

        # Crop joints and bbox
        x -= x_min
        y -= y_min
        bbox = np.array([0, 0, x_max - x_min, y_max - y_min])

        # Scale to desired size
        side_length = max(w, h)
        f_xy = float(length) / float(side_length)
        image, bbox, x, y = ImageTransformer.scale(image, bbox, x, y, f_xy)

        # Pad
        new_w, new_h = image.shape[1], image.shape[0]
        cropped = np.zeros((length, length, image.shape[2]))

        dx = length - new_w
        dy = length - new_h
        x_min, y_min = int(dx / 2.), int(dy / 2.)
        x_max, y_max = x_min + new_w, y_min + new_h

        cropped[y_min:y_max, x_min:x_max, :] = image
        x += x_min
        y += y_min

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)

        return cropped, x.astype(np.int), y.astype(np.int)

    @staticmethod
    def scale(image, bbox, x, y, f_xy):
        (h, w, _) = image.shape

        w = int(w * f_xy)
        h = int(h * f_xy)
        image = resize(image, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)

        x = x * f_xy
        y = y * f_xy
        bbox = bbox * f_xy

        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)

        return image, bbox, x, y

    @staticmethod
    def flip(image, bbox, x, y):
        image = np.fliplr(image).copy()

        w = image.shape[1]
        x_min, y_min, x_max, y_max = bbox
        bbox = np.array([w - x_max, y_min, w - x_min, y_max])
        x = w - x
        x, y = ImageTransformer.swap_joints(x, y)
        return image, bbox, x, y

    @staticmethod
    def swap_joints(x, y):
        symmetric_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [12, 13]]

        for i, j in symmetric_joints:
            x[i], x[j] = x[j].copy(), x[i].copy()
            y[i], y[j] = y[j].copy(), y[i].copy()
        return x, y

    @staticmethod
    def color_jitter(image, min_jitter, max_jitter):
        image[:, :, 0] = np.clip(image[:, :, 0] * uniform(min_jitter, max_jitter), 0., 255.)
        image[:, :, 1] = np.clip(image[:, :, 1] * uniform(min_jitter, max_jitter), 0., 255.)
        image[:, :, 2] = np.clip(image[:, :, 2] * uniform(min_jitter, max_jitter), 0., 255.)
        return image

    @staticmethod
    def normalize(image, mean, std):
        image = image / 255.
        image[:, :, 0] = (image[:, :, 0] - mean[0]) / (std[0] + 1e-8)
        image[:, :, 1] = (image[:, :, 1] - mean[1]) / (std[1] + 1e-8)
        image[:, :, 2] = (image[:, :, 2] - mean[2]) / (std[2] + 1e-8)
        return image

    @staticmethod
    def to_torch(image):
        image = np.moveaxis(image, 2, 0)
        return torch.from_numpy(image).float()
