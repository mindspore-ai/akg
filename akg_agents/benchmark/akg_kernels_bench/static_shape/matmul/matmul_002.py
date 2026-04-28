import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(Model, self).__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def forward(self, A, B):
        if self.transpose_a:
            A = A.transpose(-2, -1)
        if self.transpose_b:
            B = B.transpose(-2, -1)
        return torch.matmul(A, B)


def get_inputs():
    A = torch.randn(1024, 1344, dtype=torch.float32)
    B = torch.randn(1344, 6144, dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    transpose_a = False
    transpose_b = False
    return [transpose_a, transpose_b]