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
    def __init__(self, target_shape=(32, 1, 37)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape multiple contiguous axes: (1184) -> (32, 1, 37)
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Medium shape case: (1184) -> (32, 1, 37) = 1184 elements
    # Using fp16 data type for half precision
    input_tensor = torch.randn(1184, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape multiple contiguous axes: (1184) -> (32, 1, 37)
    target_shape = (32, 1, 37)  # Reshape multiple dimensions into 2D
    return [target_shape]
