import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise multiplication with vertical broadcast (2D).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        return input1 * input2


def get_inputs():
    input1 = torch.randn(2048, 16, dtype=torch.float16)
    input2 = torch.randn(2048, 1, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    return []


