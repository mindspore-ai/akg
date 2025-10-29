import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.argmax(input, dim, keepdim=False)
        # Returns the indices of the maximum values of all elements in the input tensor
        # or along a dimension if specified.
        # This operation is commonly used in neural networks for:
        # - Converting logits to class predictions
        # - Finding the most probable token in sequence generation
        # - Implementing hard attention mechanisms
        return torch.argmax(input_tensor, dim=self.dim)


def get_inputs():
    # Batch size: 2048
    # Hidden dimension: 8192
    input_tensor = torch.randn(2048, 8192, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for reduction
    # Reduce along second dimension (features dimension)
    dim = 1
    return [dim]
