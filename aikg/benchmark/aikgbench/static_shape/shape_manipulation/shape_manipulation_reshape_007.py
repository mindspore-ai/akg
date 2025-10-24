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
    def __init__(self, target_shape=(32768,)):
        super(Model, self).__init__()
        self.target_shape = target_shape

    def forward(self, input_tensor):
        # Reshape operation using input_tensor as input
        # Reshape multiple axes into single axis: (32, 32, 32) -> (32768, 1)
        result = torch.reshape(input_tensor, self.target_shape)
        return result


def get_inputs():
    # Small shape case: (32, 32, 32) -> (32768, 1)
    # Using fp16 data type for half precision with small shape
    input_tensor = torch.randn(32, 32, 32, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Reshape operation
    # Reshape multiple axes into single axis: (32, 32, 32) -> (32768, 1)
    target_shape = (32768, 1)  # Reshape multiple dimensions into 1D
    return [target_shape]
