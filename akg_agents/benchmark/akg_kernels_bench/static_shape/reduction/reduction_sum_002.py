import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.sum(input, dim, keepdim=False, dtype=None)
        # Reduce along first dimension (batch dimension)
        return torch.sum(input_tensor, dim=0)


def get_inputs():
    # Batch size: 256
    # Hidden dimension: 512
    input_tensor = torch.randn(256, 512, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters required
    return []