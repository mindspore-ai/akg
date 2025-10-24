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
    def __init__(self, target_shape=(64, 4096, 64)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape multiple non-contiguous axes: (64, 64, 64, 64) -> (64, 4096, 64)
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Large shape case: (64, 64, 64, 64) -> (64, 4096, 64) = 16,777,216 elements
    # Using fp32 data type for full precision
    input_tensor = torch.randn(64, 64, 64, 64, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape multiple non-contiguous axes: (64, 64, 64, 64) -> (64, 4096, 64)
    target_shape = (64, 4096, 64)  # Reshape multiple dimensions into 3D
    return [target_shape]
