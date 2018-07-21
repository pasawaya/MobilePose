
import numpy as np
from skimage.draw import circle, line
from skimage.io import imshow
from matplotlib import pyplot as plt


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
    i = 0
    for p_x, p_y, p_vis in zip(x, y, vis):
        if p_vis:
            rr, cc = circle(p_y, p_x, 4)
            image[rr, cc] = (255, 0, 0) if i == 6 else (255, 255, 255)
        i += 1

    for i, j in limbs:
        if vis[i] and vis[j]:
            rr, cc = line(y[i], x[i], y[j], x[j])
            image[rr, cc] = (0, 255, 0)

    imshow(image)
    plt.show()
