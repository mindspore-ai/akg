import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, value=1.0):
        super(Model, self).__init__()
        self.value = value

    def forward(self, input_tensor, tensor1, tensor2):
        # torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
        # Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by value, and adds it to input.
        # This operation is commonly used in neural networks for:
        # - Implementing specific mathematical formulas
        # - Attention mechanisms
        # - Fusion operations
        return torch.addcmul(input_tensor, tensor1, tensor2, value=self.value)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    tensor1 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor2 = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor, tensor1, tensor2]


def get_init_inputs():
    # Parameters for addcmul
    value = 1.0  # Scale factor
    return [value]