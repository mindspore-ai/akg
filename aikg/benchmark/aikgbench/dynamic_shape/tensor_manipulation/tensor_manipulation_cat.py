import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, inputs):
        # torch.cat(tensors, dim=0, *, out=None)
        # Concatenates the given sequence of tensors in the given dimension.
        # This operation is commonly used in neural networks for:
        # - Combining feature maps
        # - Merging batch dimensions
        # - Implementing skip connections
        return torch.cat(inputs, dim=-1)


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
    # No parameters needed for cat
    return []