import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # torch.zeros_like creates a tensor filled with zeros with the same shape as input
        # This operation is commonly used in neural networks for:
        # - Creating zero tensors with the same shape as input
        # - Initializing gradients to zero
        # - Creating masks with the same dimensions as input
        return torch.zeros_like(x, dtype=x.dtype)


def get_inputs():
    # Create input tensor with shape (128, 1024, 256)
    x = torch.randint(0, 1024, (128, 1024, 1024), dtype=torch.int64)
    return [x]


def get_init_inputs():
    return []
