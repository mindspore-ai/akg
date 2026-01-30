import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.ne(input, other, *, out=None)
        # Computes element-wise inequality.
        # Returns a boolean tensor with the same shape as input,
        # where each element is True if the corresponding elements of input and other are not equal, False otherwise.
        # This operation is commonly used in neural networks for:
        # - Implementing masking operations
        # - Comparing tensors for inequality
        # - Creating conditional masks
        return torch.ne(input1, input2)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input1 = torch.randn(1024, 4096, dtype=torch.float32)
    input2 = torch.randn(1024, 4096, dtype=torch.float32)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for ne
    return []