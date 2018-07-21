
import torch.nn as nn


def initialize_weights_normal(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        layer.bias.data.fill_(0.0)


def initialize_weights_kaiming(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        layer.bias.data.fill_(0.0)


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)