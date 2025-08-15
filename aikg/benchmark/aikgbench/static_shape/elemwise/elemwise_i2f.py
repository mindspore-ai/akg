import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(Model, self).__init__()
        self.dtype = dtype

    def forward(self, input_tensor):
        # torch.to(input, dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
        # Converts the input tensor to the specified dtype.
        # This operation is commonly used in neural networks for:
        # - Converting between different data types (e.g., int to float)
        # - Ensuring compatibility between operations that require specific dtypes
        # - Implementing mixed precision training
        return input_tensor.to(dtype=self.dtype)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Starting with integer tensor and converting to float
    input_tensor = torch.randint(0, 100, (1024, 4096), dtype=torch.int32)
    return [input_tensor]


def get_init_inputs():
    # Specific dtype for conversion
    dtype = torch.float32  # Convert to float32
    return [dtype]