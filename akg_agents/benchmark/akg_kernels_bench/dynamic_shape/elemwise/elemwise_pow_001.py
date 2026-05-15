import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, exponent):
        # torch.pow(input, exponent, *, out=None)
        # Takes the power of each element in input with exponent and returns a tensor with the result.
        # This is a power operation with exponent=2.0 (squaring).
        # Power operations are commonly used in neural networks for:
        # - Implementing polynomial activation functions
        # - Computing distance metrics
        # - Mathematical transformations in specialized layers
        return torch.pow(input_tensor, exponent)


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input_tensor1 = torch.randn(256, 512, dtype=torch.float32)
    exponent1 = torch.full((256, 512), 2.0, dtype=torch.float32)

    # Case 2: Middle (batch=1024, hidden=4096)
    input_tensor2 = torch.randn(1024, 4096, dtype=torch.float32)
    exponent2 = torch.full((1024, 4096), 2.0, dtype=torch.float32)

    # Case 3: Large (batch=2048, hidden=4096)
    input_tensor3 = torch.randn(2048, 4096, dtype=torch.float32)
    exponent3 = torch.full((2048, 4096), 2.0, dtype=torch.float32)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input_tensor4 = torch.randn(768, 2688, dtype=torch.float32)
    exponent4 = torch.full((768, 2688), 2.0, dtype=torch.float32)

    return [
        [input_tensor1, exponent1],
        [input_tensor2, exponent2],
        [input_tensor3, exponent3],
        [input_tensor4, exponent4]
    ]


def get_init_inputs():
    # No parameters needed for pow
    return []