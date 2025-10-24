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
    # Medium shape case: (56, 256, 2048) -> (56, 2048, 256)
    # Using float16 data type for half precision
    input_tensor = torch.randn(56, 256, 2048, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Transpose operation
    # Transpose 3D tensor: (56, 256, 2048) -> (56, 2048, 256)
    dim0 = 1
    dim1 = 2
    return [dim0, dim1]
