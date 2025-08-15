import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.log(input, *, out=None)
        # Computes the element-wise natural logarithm of the input tensor.
        # This operation is commonly used in neural networks for:
        # - Implementing logarithmic functions in probability computations
        # - Loss functions (e.g., negative log-likelihood)
        # - Mathematical transformations
        return torch.log(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Using positive values since log is undefined for negative values
    input_tensor = torch.rand(1024, 4096, dtype=torch.float32) + 0.1  # Adding 0.1 to ensure positive values
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for log
    return []