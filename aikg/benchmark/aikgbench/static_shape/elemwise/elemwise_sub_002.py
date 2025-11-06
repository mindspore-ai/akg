import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise subtraction (3D, bfloat16).
    Large scale: e8
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D subtraction operation, broadcasting on front and middle dimensions
        return input1 - input2


def get_inputs():

    input1 = torch.randint(-128, 127, (65536, 128, 16), dtype=torch.int32)
    input2 = torch.randint(-128, 127, (1, 1, 16), dtype=torch.int32)
    return [input1, input2]


def get_init_inputs():
    return []


