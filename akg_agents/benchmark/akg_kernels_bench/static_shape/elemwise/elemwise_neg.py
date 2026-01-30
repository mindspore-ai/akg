import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.neg(input, *, out=None)
        # Returns a new tensor with the negative of the elements of input.
        # This operation is commonly used in neural networks for:
        # - Implementing mathematical transformations
        # - Computing differences or residuals
        # - Implementing certain activation functions
        return torch.neg(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for neg
    return []