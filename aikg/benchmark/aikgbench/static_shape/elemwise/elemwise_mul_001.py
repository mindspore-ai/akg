import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise multiplication operation (3D, int8).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D multiplication operation
        result = input1 * input2
        return result

def get_inputs():
    # Medium scale: 32 * 512 * 1024 ≈ e7

    input1 = torch.randint(-128, 127, (32, 512, 1), dtype=torch.int8)
    input2 = torch.randint(-128, 127, (32, 512, 1024), dtype=torch.int8)
    return [input1, input2]

def get_init_inputs():
    # No parameters needed for multiplication
    return []