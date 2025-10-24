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
    def __init__(self, target_shape=(32, 2, 4096)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape from 2D (64, 4096) to 3D (32, 2, 4096) tensor
        # This operation changes the tensor layout while preserving total elements (262,144)
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Small shape case: (64, 4096) -> (32, 2, 4096) = 262,144 elements
    # Input tensor with 64 samples, each with 4096 features
    # Using bfloat16 data type for memory efficiency and numerical stability
    input_tensor = torch.randn(64, 4096, dtype=torch.bfloat16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape from 2D to 3D: (64, 4096) -> (32, 2, 4096)
    # This creates a 3D tensor with 32 batches, 2 groups, and 4096 features per group
    target_shape = (32, 2, 4096)  # Reshape the last dimension into 3D structure
    return [target_shape]
