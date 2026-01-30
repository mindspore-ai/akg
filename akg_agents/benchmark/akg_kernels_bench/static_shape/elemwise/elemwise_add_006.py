import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise addition (3D, FP16) with broadcasting on front and back dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D addition operation, broadcasting on front and back dimensions
        return input1 + input2


def get_inputs():
    input1 = torch.randn(16, 1024, 65536, dtype=torch.float16)
    input2 = torch.randn(1, 1024, 1, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for add
    return []


