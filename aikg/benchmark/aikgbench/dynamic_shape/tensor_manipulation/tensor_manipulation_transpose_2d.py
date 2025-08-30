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


def get_inputs_dyn_list():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Three tensors are concatenated along the batch dimension to get (1536, 4096)
    # Small shape case
    inputs1 = torch.randn(128, 256, dtype=torch.float32)
    # Non-aligned shape case
    inputs2 = torch.randn(511, 511, dtype=torch.float32)
    # Middle shape case
    inputs3 = torch.randn(512, 4096, dtype=torch.float32)
    # Standard Large shape case
    inputs4 = torch.randn(1024, 4096, dtype=torch.float32)
    # Large shape case
    inputs5 = torch.randn(2048, 8192, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
        [inputs5]
    ]


def get_init_inputs():
    # No parameters needed for transpose
    return []