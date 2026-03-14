import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b, bias):
        return torch.matmul(a, b) + bias


def get_inputs():
    a = torch.randn(4096, 4096, dtype=torch.float16)
    b = torch.randn(4096, 4096, dtype=torch.float16)
    bias = torch.randn(1, 4096, dtype=torch.float16)
    return [a, b, bias]


def get_init_inputs():
    return []
