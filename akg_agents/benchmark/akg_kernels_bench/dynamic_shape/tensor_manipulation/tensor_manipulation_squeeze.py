import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.squeeze(input, dim=None, out=None)
        # Returns a tensor with all the dimensions of input of size 1 removed.
        # This operation is commonly used in neural networks for:
        # - Removing unnecessary dimensions
        # - Cleaning up tensor shapes after operations
        # - Ensuring compatibility with other operations
        return torch.squeeze(input_tensor)


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(128, 256, 1, dtype=torch.float32)

    # Middle shape case
    input2 = torch.randn(512, 1024, 1, dtype=torch.float32)

    # Large shape case
    input3 = torch.randn(1024, 4096, 1, dtype=torch.float32)

    # Noaligned shape case
    input4 = torch.randn(511, 3000, 1, dtype=torch.float32)

    return [[input1], [input2], [input3], [input4]]


def get_init_inputs():
    # No parameters needed for squeeze
    return []