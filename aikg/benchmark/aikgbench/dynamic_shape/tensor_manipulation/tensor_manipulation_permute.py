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


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(256, 128, 64, dtype=torch.float32)

    # Middle shape case
    input2 = torch.randn(1024, 1024, 256, dtype=torch.float32)

    # Large shape case
    input3 = torch.randn(2048, 4096, 512, dtype=torch.float32)

    # Noaligned shape case
    input4 = torch.randn(513, 3000, 384, dtype=torch.float32)

    return [[input1], [input2], [input3], [input4]]


def get_init_inputs():
    # Specific dims for permutation
    # Permute dimensions to (128, 4, 64)
    dims = (1, 0, 2)
    return [dims]