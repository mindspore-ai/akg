import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, input_tensor):
        # tensor.to(dtype)
        # Casts the tensor to the specified dtype using to() method.
        # This operation is commonly used in neural networks for:
        # - Converting from int8 to float32 for higher precision
        # - Expanding precision for mathematical operations
        # - Preparing data for neural network layers
        return input_tensor.to(self.dtype)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 1024
    input_tensor = torch.randint(-128, 127, (1024, 1024), dtype=torch.int8)  # Starting with int8
    return [input_tensor]


def get_init_inputs():
    # Parameters for cast - target dtype
    dtype = torch.float32  # Target dtype: int8 -> float32
    return []

