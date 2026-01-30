import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var):
        # torch.argmax(input, dim, keepdim=False)
        # Returns the indices of the maximum values of all elements in the input tensor
        # or along a dimension if specified.
        # This operation is commonly used in neural networks for:
        # - Converting logits to class predictions
        # - Finding the most probable token in sequence generation
        # - Implementing hard attention mechanisms
        return torch.argmax(var, dim=1)


def get_inputs_dyn_list():
    # Argmax reduction variation cases with both aligned and non-aligned shapes

    # Case 1: Large tensor size 512x512 (aligned)
    inputs1 = torch.randn(512, 512, dtype=torch.float32)

    # Case 2: Very large tensor size 1023x1023 (non-aligned)
    inputs2 = torch.randn(1023, 1023, dtype=torch.float32)

    # Case 3: Very large tensor size 1024x1024 (aligned)
    inputs3 = torch.randn(1024, 1024, dtype=torch.float32)

    # Case 4: Extreme tensor size 4096x4096 (aligned)
    inputs4 = torch.randn(4096, 4096, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
    ]


def get_init_inputs():
    # No parameters needed for argmax
    return []
