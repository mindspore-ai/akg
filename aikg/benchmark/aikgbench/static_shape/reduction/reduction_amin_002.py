import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.amin(input, dim, keepdim=False)
        # Returns the minimum value of each row of the input tensor in the given dimension(s) dim.
        # This function is very similar to torch.min(), but differs in how gradients are propagated.
        # It is commonly used in neural networks for:
        # - Robust pooling operations
        # - Finding extreme values in feature maps
        return torch.amin(input_tensor, self.dim)


def get_inputs():
    # Batch size: 16
    # Hidden dimension: 262144
    input_tensor = torch.randn(16, 262144, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for reduction
    # Reduce along second dimension (features dimension)
    dim = 1
    return [dim]