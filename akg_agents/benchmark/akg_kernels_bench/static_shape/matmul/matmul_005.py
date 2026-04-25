import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, transpose_a=False, transpose_b=True):
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
    A = torch.randn(1024, 2688, dtype=torch.bfloat16)
    B = torch.randn(5120, 2688, dtype=torch.bfloat16)
    return [A, B]


def get_init_inputs():
    transpose_a = False
    transpose_b = True
    return [transpose_a, transpose_b]