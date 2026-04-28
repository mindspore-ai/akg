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
    def __init__(self, perm=[0, 2, 1]):
        super(Model, self).__init__()
        self.perm = perm

    def forward(self, input_tensor):
        # Transpose operation using input_tensor as input
        result = torch.permute(input_tensor, self.perm)
        return result

def get_inputs():
    # Batch size: 32
    # Height: 128
    # Width: 256
    input_tensor = torch.randn(32, 128, 256, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    # Parameters for Transpose operation
    perm = [0, 2, 1]  # Permutation of dimensions
    return [perm]