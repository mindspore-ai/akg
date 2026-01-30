import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.bitwise_not(input, *, out=None)
        # Computes the element-wise bitwise NOT of the given input tensor.
        # This operation is commonly used in neural networks for:
        # - Implementing bit manipulation operations
        # - Creating bit masks
        # - Low-level data processing
        return torch.bitwise_not(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randint(0, 256, (1024, 4096), dtype=torch.int32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for bitwise_not
    return []