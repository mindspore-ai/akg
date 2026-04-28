import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x0, x1):
        ret = x0 + x1
        ret = ret + x1  
        ret = ret + x1
        ret = ret + x0  
        ret = ret + x0
        ret = ret + x0
        return ret

def get_inputs():
    # Create input tensors with huge shapes for extreme stress testing
    # Shape: (1024, 2048, 4096) = 8B elements
    L, M, N = 2, 256, 16
    x0 = torch.randn(L, M, N, dtype=torch.int32)
    x1 = torch.randn(L, M, N, dtype=torch.int32)
    return [x0, x1]


def get_init_inputs():
    return []
