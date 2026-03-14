import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


def get_inputs():
    a = torch.randn(32, 8192, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16)
    return [a, b]


def get_init_inputs():
    return []
