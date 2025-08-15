import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Mean along dimension 0 (batch dimension)
        return torch.mean(input_tensor, 0)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters required
    return []