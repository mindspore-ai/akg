import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.tan(input, *, out=None)
        # Returns a new tensor with the tangent of the elements of input.
        # This operation is commonly used in neural networks for:
        # - Implementing certain activation functions
        # - Mathematical transformations in specialized layers
        # - Periodic function approximations
        return torch.tan(input_tensor)


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input_tensor1 = torch.randn(256, 512, dtype=torch.float32)

    # Case 2: Middle (batch=1024, hidden=4096)
    input_tensor2 = torch.randn(1024, 4096, dtype=torch.float32)

    # Case 3: Large (batch=2048, hidden=4096)
    input_tensor3 = torch.randn(2048, 4096, dtype=torch.float32)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input_tensor4 = torch.randn(768, 2688, dtype=torch.float32)

    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]


def get_init_inputs():
    # No parameters needed for tan
    return []