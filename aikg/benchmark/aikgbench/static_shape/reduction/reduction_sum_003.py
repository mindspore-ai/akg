import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Sum over all elements
        return torch.sum(input_tensor)


def get_inputs():
    input_tensor = torch.randn(2048, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters required
    return []