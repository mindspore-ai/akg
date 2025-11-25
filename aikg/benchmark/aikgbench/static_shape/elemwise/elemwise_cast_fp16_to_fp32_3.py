import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, input_tensor):
        # tensor.to(dtype)
        # Casts the tensor to the specified dtype using tensor.to() method.
        # This operation is commonly used in neural networks for:
        # - Converting from float16 to float32 for higher precision
        # - Restoring precision after mixed precision training
        # - Ensuring numerical stability in critical computations
        return torch.cast(input_tensor, self.dtype)


def get_inputs():
    # Batch size: 1536 (large batch)
    # Hidden dimension: 6144 (large hidden dim)
    input_tensor = torch.randn(128, 1024, 1024, dtype=torch.float16)  # Starting with float16
    return [input_tensor]


def get_init_inputs():
    # Parameters for cast - target dtype
    dtype = torch.float32  # Target dtype: float16 -> float32
    return []
