import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.zeros creates a tensor filled with zeros
        # This operation is commonly used in neural networks for:
        # - Initializing weight matrices
        # - Creating zero padding tensors
        # - Generating reference tensors for testing
        device = torch.device("npu")
        return torch.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=device)


def get_inputs():
    input_tensor = torch.randn(128, 1024, 1024, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    return []
