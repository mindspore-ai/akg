import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise addition operation (3D).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D addition operation
        result = input1 + input2
        return result

def get_inputs():
    # Medium scale: 32 * 512 * 1024 ≈ e7

    input1 = torch.randn(32, 512, 1024, dtype=torch.float32)
    input2 = torch.randn(32, 512, 1024, dtype=torch.float32)
    return [input1, input2]

def get_init_inputs():
    # No parameters for add
    return []