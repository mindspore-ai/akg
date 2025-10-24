import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Exponential activation function operation (3D, FP32).
    Large scale: e7
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D exponential activation function
        result = torch.exp(input_tensor)
        return result

def get_inputs():
    # Medium scale: 32 * 512 * 1024 ≈ e7

    input_tensor = torch.randn(32, 512, 1024, dtype=torch.float32) * 0.1  # Scale down to prevent overflow
    return [input_tensor]

def get_init_inputs():
    # No parameters for Exponential activation operation
    return []