import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Sum over all elements
        return torch.sum(input_tensor)


def get_inputs():
    # Batch size: 256
    # Hidden dimension: 512
    input_tensor = torch.randn(256, 512, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters required
    return []