import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Leaky ReLU activation function operation.
    This operation is commonly used in neural networks for:
    - Activation function that addresses "dying ReLU" problem
    - Allows small negative values when input is negative
    - Used in some deep learning architectures
    
    Formula: output = max(input, negative_slope * input)
    """
    def __init__(self, negative_slope=0.01):
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input_tensor):
        # Leaky ReLU activation function applied to input_tensor
        result = torch.nn.functional.leaky_relu(input_tensor, negative_slope=self.negative_slope)
        return result

def get_inputs():
    # Batch size: 32
    # Sequence length: 512
    # Hidden size: 1024
    input_tensor = torch.randn(32, 512, 1024, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    # Parameters for Leaky ReLU activation operation
    negative_slope = 0.01
    return [negative_slope]