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
    # Medium shape case: (63, 512, 1024) -> (1024, 63, 512)
    # Using float16 data type for half precision
    input_tensor = torch.randn(56, 256, 2048, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # Parameters for Permute operation
    # Permute 3D tensor: (63, 512, 1024) -> (1024, 63, 512)
    dims = (2, 0, 1)
    return [dims]
