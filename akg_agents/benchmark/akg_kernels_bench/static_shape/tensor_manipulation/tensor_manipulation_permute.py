import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dims=None):
        super(Model, self).__init__()
        self.dims = dims

    def forward(self, input_tensor):
        # torch.permute(input, dims)
        # Returns a view of the original tensor input with its dimensions reordered.
        # This operation is commonly used in neural networks for:
        # - Changing tensor layouts for compatibility with other operations
        # - Implementing transpose operations
        # - Rearranging dimensions for specific computations
        return torch.permute(input_tensor, dims=self.dims)


def get_inputs():
    # Batch size: 4
    # Sequence length: 128
    # Hidden dimension: 64
    input_tensor = torch.randn(4, 128, 64, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dims for permutation
    # Permute dimensions to (128, 4, 64)
    dims = (1, 0, 2)
    return [dims]