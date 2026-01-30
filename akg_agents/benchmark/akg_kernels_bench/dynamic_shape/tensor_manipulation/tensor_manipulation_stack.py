import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.stack(tensors, dim=0, *, out=None)
        # Concatenates a sequence of tensors along a new dimension.
        # This operation is commonly used in neural networks for:
        # - Combining multiple feature maps into a new dimension
        # - Creating batch dimensions
        # - Implementing certain attention mechanisms
        return torch.stack([input1, input2], dim=0)


def get_inputs_dyn_list():
    # Two tensors are stacked along dimension 0 to create a new batch dimension
    # Small shape case - stack along batch dimension
    inputs1_1 = torch.randn(128, 256, dtype=torch.float32)
    inputs1_2 = torch.randn(128, 256, dtype=torch.float32)
    # Non-aligned shape case - stack along batch dimension
    inputs2_1 = torch.randn(511, 511, dtype=torch.float32)
    inputs2_2 = torch.randn(511, 511, dtype=torch.float32)
    # Middle shape case - stack along batch dimension
    inputs3_1 = torch.randn(512, 4096, dtype=torch.float32)
    inputs3_2 = torch.randn(512, 4096, dtype=torch.float32)
    # Standard Large shape case - stack along batch dimension
    inputs4_1 = torch.randn(1024, 4096, dtype=torch.float32)
    inputs4_2 = torch.randn(1024, 4096, dtype=torch.float32)
    # Large shape case - stack along batch dimension
    inputs5_1 = torch.randn(2048, 8192, dtype=torch.float32)
    inputs5_2 = torch.randn(2048, 8192, dtype=torch.float32)

    return [
        [inputs1_1, inputs1_2],
        [inputs2_1, inputs2_2],
        [inputs3_1, inputs3_2],
        [inputs4_1, inputs4_2],
        [inputs5_1, inputs5_2]
    ]


def get_init_inputs():
    # No parameters needed for stack
    return []