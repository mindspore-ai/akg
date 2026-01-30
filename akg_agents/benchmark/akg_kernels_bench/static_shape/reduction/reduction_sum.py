import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.sum(input, dim, keepdim=False, dtype=None)
        # Returns the sum of each row of the input tensor in the given dimension dim.
        # This is commonly used in neural networks for:
        # - Computing loss over a batch or aggregating features
        # - Reducing tensor dimensions
        return torch.sum(input_tensor, 1)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters required
    return []