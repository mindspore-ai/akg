import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.log2(input, *, out=None)
        # Returns a new tensor with the logarithm to the base 2 of the elements of input.
        # This operation is commonly used in neural networks for:
        # - Implementing certain loss functions
        # - Mathematical transformations in specialized layers
        # - Computing information-theoretic quantities (bits)
        return torch.log2(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Using positive values for log2 since log of non-positive numbers is not defined
    input_tensor = torch.rand(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to ensure positive values
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for log2
    return []