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
    def __init__(self, target_shape=(64, 128)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape the last axis (feature dimension) from 256 to 64x4
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Small shape case: (32, 256) -> (64, 128) = 8,192 elements
    # Using int8 data type for memory efficiency
    input_tensor = torch.randint(-128, 127, (32, 256), dtype=torch.int8)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape last axis: (32, 256) -> (64, 128)
    target_shape = (64, 128)  # Reshape the last dimension
    return [target_shape]
