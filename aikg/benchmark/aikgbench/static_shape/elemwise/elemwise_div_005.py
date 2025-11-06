import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        return dividend / divisor


def get_inputs():
    dividend = torch.randn(65536, 128, 16, dtype=torch.float32)
    divisor = torch.randn(1, 128, 1, dtype=torch.float32) + 1.0
    return [dividend, divisor]


def get_init_inputs():
    return []

