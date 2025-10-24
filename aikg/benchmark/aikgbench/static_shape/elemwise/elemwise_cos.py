import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.cos(input, *, out=None)
        # Computes the element-wise cosine of the input tensor.
        # This operation is commonly used in neural networks for:
        # - Implementing periodic activation functions
        # - Positional encoding in transformers
        # - Signal processing operations
        return torch.cos(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for cos
    return []