
import numpy as np
from skimage.transform import resize
from random import random


class ImageTransformer(object):
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, x, y, visibility):

        x_min, x_max = max(0, np.amin(x)), min(image.shape[1] - 1, np.amax(x))
        y_min, y_max = max(0, np.amin(y)), min(image.shape[0] - 1, np.amax(y))
        bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)

        # Scale
        f_xy = self.min_scale + (self.max_scale - self.min_scale) * random()
        image, bbox, x, y = ImageTransformer.scale(image, bbox, x, y, f_xy)
        return image

    @staticmethod
    def scale(image, bbox, x, y, f_xy):
        (h, w, _) = image.shape

        w = int(w * f_xy)
        h = int(h * f_xy)
        image = resize(image, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)

        x = x * f_xy
        y = y * f_xy
        bbox = bbox * f_xy

        # WRONG, don't clip, mark visibility 0
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)

        return image, bbox.astype(np.int32), x.astype(np.int32), y.astype(np.int32)
