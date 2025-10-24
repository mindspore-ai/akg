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
    # Large shape case: (1024, 512, 256) -> (256, 512, 1024)
    # Using float32 data type for full precision
    input_tensor = torch.randn(1024, 512, 256, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Permute operation
    # Permute 3D tensor: (1024, 512, 256) -> (256, 512, 1024)
    dims = (2, 1, 0)
    return [dims]
