import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dtype=torch.bool):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, input_tensor):
        # torch.to(input, dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
        # Casts the input tensor to the specified dtype.
        # This operation is commonly used in neural networks for:
        # - Converting between different data types (e.g., float to bool)
        # - Ensuring compatibility between operations that require specific dtypes
        # - Implementing mixed precision training
        return input_tensor.to(dtype=self.dtype)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Starting with float tensor and converting to bool
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dtype for conversion
    dtype = torch.bool  # Convert to boolean
    return [dtype]