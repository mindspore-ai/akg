import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, shape=(64, 64, 1024)):
        super(Model, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        # torch.reshape(input, shape)
        # Returns a tensor with the same data and number of elements as input, but with the specified shape.
        # This operation is commonly used in neural networks for:
        # - Changing tensor shapes for specific operations
        # - Implementing certain matrix operations
        # - Reshaping tensors for compatibility with other operations
        return torch.reshape(input_tensor, self.shape)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Reshaping to (64, 64, 1024) for specific operations
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters needed for reshape
    shape = (64, 64, 1024)  # Reshape dimensions
    return [shape]