import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Transpose operation that transposes the dimensions of the input tensor.
    This operation is commonly used in neural networks for:
    - Changing the order of tensor dimensions
    - Used in various tensor manipulation tasks
    - Converting between different data layouts
    
    Formula: Reorders the dimensions of the input tensor according to the permutation
    """
    def __init__(self, dim0, dim1):
        super(Model, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input_tensor):
        result = torch.transpose(input_tensor, self.dim0, self.dim1)
        return result

def get_inputs():
    # Large shape case: (16, 32, 64, 128) -> (16, 64, 32, 128) = 4,194,304 elements
    # Using bfloat16 data type for memory efficiency and numerical stability
    input_tensor = torch.randn(16, 32, 64, 128, dtype=torch.bfloat16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Transpose operation
    # Transpose 4D tensor: (16, 32, 64, 128) -> (16, 64, 32, 128)
    dim0 = 1
    dim1 = 2
    return [dim0, dim1]
