
import torch.nn as nn
import os
import torch
import shutil


def save_checkpoint(state, is_best, checkpoint, best_name='best'):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_name + '.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, best_name='best'):
    filepath = os.path.join(checkpoint, best_name)
    if not os.path.exists(filepath):
        raise("File doesn't exist {}".format(filepath))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


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