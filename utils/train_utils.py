
import torch.nn as nn
import torch.onnx
import onnx_coreml

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


def save_coreml(model, dummy_input, model_name):
    onnx_model_name = 'model.onnx'
    torch.onnx.export(model, dummy_input, onnx_model_name)
    mlmodel = onnx_coreml.convert(onnx_model_name,
                                  mode='regressor',
                                  image_input_names='0',
                                  image_output_names='309',
                                  predicted_feature_name='keypoints')
    mlmodel.save(model_name)


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)