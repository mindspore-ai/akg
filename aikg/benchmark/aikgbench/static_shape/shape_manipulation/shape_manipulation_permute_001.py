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
    # Small shape case: (32, 256) -> (256, 32) = 8,192 elements
    # Using int8 data type for memory efficiency
    input_tensor = torch.randint(-128, 127, (32, 256), dtype=torch.int8)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Permute operation
    # Permute 2D tensor: (32, 256) -> (256, 32)
    dims = (1, 0)
    return [dims]
