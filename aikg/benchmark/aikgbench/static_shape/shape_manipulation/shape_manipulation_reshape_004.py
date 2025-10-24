import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Reshape operation that changes the shape of the input tensor.
    This operation is commonly used in neural networks for:
    - Changing tensor dimensions for specific operations
    - Implementing certain matrix operations
    - Reshaping tensors for compatibility with other operations
    
    Formula: Reshapes input tensor to the specified shape while preserving total elements
    """
    def __init__(self, target_shape=(1024, 32, 32)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape from 4D (32, 32, 32, 32) to 3D (1024, 32, 32) tensor
        # This flattens the 4D tensor into a 3D matrix while preserving total elements (1,048,576)
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Medium shape case: (32, 32, 32, 32) -> (1024, 32, 32) = 1,048,576 elements
    # Input tensor with 4D structure: 32x32x32x32 = 1,048,576 total elements
    # Using bfloat16 data type for memory efficiency and numerical stability
    input_tensor = torch.randn(32, 32, 32, 32, dtype=torch.bfloat16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape from 4D to 2D: (32, 32, 32, 32) -> (1024, 32, 32)
    # This flattens the 4D tensor into a square 3D matrix (1024x32x32)
    target_shape = (1024, 32, 32)  # Reshape multiple dimensions into 3D square matrix
    return [target_shape]
