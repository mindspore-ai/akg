import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hard Sigmoid activation function operation.
    This operation is commonly used in neural networks for:
    - Efficient activation function for mobile and edge devices
    - Used in MobileNetV3 and other efficient architectures
    - Provides a piecewise linear approximation to Sigmoid
    
    Formula: output = relu6(input + 3) / 6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Hard Sigmoid activation function applied to input_tensor
        result = torch.nn.functional.hardsigmoid(input_tensor)
        return result

def get_inputs():
    # Batch size: 32
    # Sequence length: 512
    # Hidden size: 1024
    input_tensor = torch.randn(32, 512, 1024, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    # No parameters for Hard Sigmoid activation operation
    return []