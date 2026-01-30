import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, layer_shape):
        super(Model, self).__init__()
        self.layer_shape = layer_shape

    def forward(self, inp, residual, weight, bias):
        # Fused operation: add_layer_norm
        # Computes layer_norm(inp + residual, layer_shape, weight, bias) in a single operation
        # More efficient than computing add and layer_norm separately
        added = inp + residual
        return torch.layer_norm(added, self.layer_shape, weight, bias)


def get_inputs():
    shape = (1024, 4096)
    inp = torch.randn(shape, dtype=torch.float32)
    residual = torch.randn(shape, dtype=torch.float32)
    weight = torch.randn(shape[-1], dtype=torch.float32)
    bias = torch.randn(shape[-1], dtype=torch.float32)
    return [inp, residual, weight, bias]


def get_init_inputs():
    shape = (1024, 4096)
    layer_shape = (shape[-1],)
    return [layer_shape]


