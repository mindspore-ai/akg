import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # torch.div(input, other, *, rounding_mode='trunc', out=None)
        # Performs division with truncation towards zero.
        # This is equivalent to C-style integer division.
        # This operation is commonly used in neural networks for:
        # - Implementing ceiling division operations
        # - Calculating grid dimensions in CUDA kernels
        # - Mathematical transformations that require integer division
        return torch.div(dividend, divisor, rounding_mode='trunc')


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    dividend1 = torch.randn(256, 512, dtype=torch.float32)
    divisor1 = torch.randn(256, 512, dtype=torch.float32) + 1e-6

    # Case 2: Middle (batch=1024, hidden=4096)
    dividend2 = torch.randn(1024, 4096, dtype=torch.float32)
    divisor2 = torch.randn(1024, 4096, dtype=torch.float32) + 1e-6

    # Case 3: Large (batch=2048, hidden=4096)
    dividend3 = torch.randn(2048, 4096, dtype=torch.float32)
    divisor3 = torch.randn(2048, 4096, dtype=torch.float32) + 1e-6

    # Case 4: Non-aligned (batch=768, hidden=2688)
    dividend4 = torch.randn(768, 2688, dtype=torch.float32)
    divisor4 = torch.randn(768, 2688, dtype=torch.float32) + 1e-6

    return [
        [dividend1, divisor1],
        [dividend2, divisor2],
        [dividend3, divisor3],
        [dividend4, divisor4]
    ]


def get_init_inputs():
    # No parameters needed for cdiv
    return []