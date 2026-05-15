import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)


def get_inputs():
    A = torch.randn(2048, 7168, dtype=torch.bfloat16)
    B = torch.randn(7168, 16384, dtype=torch.bfloat16)
    return [A, B]


def get_init_inputs():
    return []
