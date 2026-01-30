import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.sqrt(input, *, out=None)
        # Returns a new tensor with the square root of the elements of input.
        # This operation is commonly used in neural networks for:
        # - Normalization operations (e.g., RMS normalization)
        # - Computing standard deviations
        # - Mathematical transformations in specialized layers
        return torch.sqrt(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Using positive values for sqrt since sqrt of negative numbers is not defined in reals
    input_tensor = torch.rand(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to ensure positive values
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for sqrt
    return []