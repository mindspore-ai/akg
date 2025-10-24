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
    def __init__(self, target_shape=(1280, 512, 8)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape the last axis (feature dimension) from 4096 to 1280x512x8
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Small shape case: (1280, 4096) -> (1280, 512, 8) = 131,072 elements
    # Using fp16 data type for half precision
    input_tensor = torch.randn(1280, 4096, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape last axis: (1280, 4096) -> (1280, 512, 8)
    target_shape = (1280, 512, 8)  # Reshape the last dimension
    return [target_shape]
