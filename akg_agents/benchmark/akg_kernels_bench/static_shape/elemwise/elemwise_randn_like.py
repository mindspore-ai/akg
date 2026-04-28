import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
        # Returns a tensor with the same size as input that is filled with random numbers from a normal distribution.
        # This operation is commonly used in neural networks for:
        # - Weight initialization
        # - Generating random noise for regularization
        # - Implementing random sampling operations
        return torch.randn_like(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for randn_like
    return []