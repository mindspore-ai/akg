import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        return input1 + input2


def get_inputs():
    input1 = torch.randn(131072, 2048, dtype=torch.float32)
    input2 = torch.randn(1, 2048, dtype=torch.float32)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for add
    return []


