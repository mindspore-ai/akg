import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.t(input)
        # Expects input to be a 2-D tensor and transposes dimensions 0 and 1.
        # This operation is commonly used in neural networks for:
        # - Transposing weight matrices in linear layers
        # - Changing tensor layouts for compatibility with other operations
        # - Implementing certain mathematical transformations
        return torch.t(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Transposing gives us (4096, 1024)
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for transpose
    return []