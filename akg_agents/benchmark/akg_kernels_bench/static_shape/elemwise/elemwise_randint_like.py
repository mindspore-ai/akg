import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.randint_like(input, low, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
        # Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly.
        # This operation is commonly used in neural networks for:
        # - Generating random indices
        # - Creating random integer tensors for operations
        # - Implementing random sampling operations
        return torch.randint_like(input_tensor, 0, 100)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for randint_like
    return []