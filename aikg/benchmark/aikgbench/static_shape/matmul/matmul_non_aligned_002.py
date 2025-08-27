import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, mat1, mat2):
        return torch.matmul(mat1, mat2)


def get_inputs():
    mat1 = torch.randn(128, 1024, dtype=torch.float16)
    mat2 = torch.randn(1024, 512, dtype=torch.float16)
    return [mat1, mat2]


def get_init_inputs():
    return []