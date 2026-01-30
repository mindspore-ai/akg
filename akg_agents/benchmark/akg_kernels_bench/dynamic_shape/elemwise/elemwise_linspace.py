import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        # Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        # This operation is commonly used in neural networks for:
        # - Creating evenly spaced values for interpolation
        # - Generating sequences with specific spacing
        # - Implementing certain mathematical transformations
        return torch.linspace(0.0, 1.0, 4096, dtype=torch.float32)


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input_tensor1 = torch.randn(256, 512, dtype=torch.float32)

    # Case 2: Middle (batch=1024, hidden=4096)
    input_tensor2 = torch.randn(1024, 4096, dtype=torch.float32)

    # Case 3: Large (batch=2048, hidden=4096)
    input_tensor3 = torch.randn(2048, 4096, dtype=torch.float32)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input_tensor4 = torch.randn(761, 1344, dtype=torch.float32)

    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]


def get_init_inputs():
    # No parameters needed for linspace
    return []