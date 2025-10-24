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
    # Large shape case: (10, 20, 30, 40) -> (10, 30, 10, 40)
    # Using bfloat16 data type for memory efficiency and numerical stability
    input_tensor = torch.randn(10, 20, 30, 40, dtype=torch.bfloat16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Permute operation
    # Permute 4D tensor: (10, 20, 30, 40) -> (10, 30, 10, 40)
    dims = (0, 2, 1, 3)
    return [dims]
