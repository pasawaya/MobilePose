
import numpy as np


def to_numpy(data):
    n = len(data)
    x, y, vis = np.zeros(n), np.zeros(n), np.zeros(n)
    for p in range(n):
        x[p] = data[str(p)][0]
        y[p] = data[str(p)][1]
        vis[p] = data[str(p)][2]
    return x, y, vis
