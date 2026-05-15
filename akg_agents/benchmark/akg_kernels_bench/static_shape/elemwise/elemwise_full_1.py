import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.full creates a tensor filled with a scalar value
        # This operation is commonly used in neural networks for:
        # - Creating constant tensors for initialization
        # - Generating masks or padding tensors
        # - Creating reference tensors for testing
        device = torch.device("npu")
        return torch.full(input_tensor, 100, dtype=input_tensor.dtype, device=device)


def get_inputs():
    input_tensor = torch.randn(2, 256, 16, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    return []
