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
    # Large shape case: (63, 128, 256) -> (256, 128, 63)
    # Using float32 data type for full precision
    input_tensor = torch.randn(63, 128, 256, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Transpose operation
    # Transpose 3D tensor: (63, 128, 256) -> (256, 128, 63)
    dim0 = 0
    dim1 = 2
    return [dim0, dim1]
