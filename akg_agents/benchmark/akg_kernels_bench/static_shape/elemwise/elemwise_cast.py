import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, input_tensor):
        # torch.tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
        # Casts the tensor to the specified dtype.
        # This operation is commonly used in neural networks for:
        # - Converting between data types for compatibility
        # - Reducing memory usage (e.g., float32 to float16)
        # - Meeting the requirements of specific operations
        return input_tensor.to(self.dtype)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float64)  # Starting with float64 to cast to float32
    return [input_tensor]


def get_init_inputs():
    # Parameters for cast
    dtype = torch.float32  # Target dtype
    return [dtype]