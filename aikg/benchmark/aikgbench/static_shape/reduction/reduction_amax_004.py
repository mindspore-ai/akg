import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.amax(input, keepdim=False)
        # Returns the maximum value of the input tensor.
        return torch.amax(input_tensor)


def get_inputs():
    # Sequence length: 16384
    input_tensor = torch.randn(16384, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    return []