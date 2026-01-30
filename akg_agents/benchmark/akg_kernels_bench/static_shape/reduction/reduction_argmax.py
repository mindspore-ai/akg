import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.argmax(input, dim, keepdim=False)
        # Returns the indices of the maximum values of all elements in the input tensor
        # or along a dimension if specified.
        # This operation is commonly used in neural networks for:
        # - Converting logits to class predictions
        # - Finding the most probable token in sequence generation
        # - Implementing hard attention mechanisms
        return torch.argmax(input_tensor, dim=1)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for argmax
    return []