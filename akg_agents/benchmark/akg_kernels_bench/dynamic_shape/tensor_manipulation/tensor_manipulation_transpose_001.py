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


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(16, 64, 512, dtype=torch.float32)
    # Middle shape case
    input2 = torch.randn(32, 512, 1024, dtype=torch.float32)
    # Large shape case
    input3 = torch.randn(256, 1024, 4096, dtype=torch.float32)
    # Nonaligned shape case
    input4 = torch.randn(48, 256, 2688, dtype=torch.float32)

    return [[input1], [input2], [input3], [input4]]

def get_init_inputs():
    # Parameters for Transpose operation
    perm = [0, 2, 1]  # Permutation of dimensions
    return [perm]