
import torch.nn as nn


# temporary
def clip_grad_norm_(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            print('clipping')
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


def initialize_weights_normal(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def initialize_weights_kaiming(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_parameters_rec(model):
    return sum(p.numel() for module in model.modules() for p in module.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)