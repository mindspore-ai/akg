import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.transpose(input, dim0, dim1)
        # Returns a tensor that is a transposed version of input.
        # This operation is commonly used in neural networks for:
        # - Changing tensor layouts
        # - Implementing certain matrix operations
        # - Reshaping tensors for specific operations
        return torch.transpose(input_tensor, -2, -1)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Transposing gives us (4096, 1024)
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for transpose
    return []