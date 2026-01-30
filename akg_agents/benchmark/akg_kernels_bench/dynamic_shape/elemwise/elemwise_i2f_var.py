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


def get_inputs_dyn_list():
    # Integer to float conversion variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inputs1 = torch.randint(0, 100, (256, 1024), dtype=torch.int32)
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inputs2 = torch.randint(0, 100, (1024, 4096), dtype=torch.int32)
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inputs3 = torch.randint(0, 100, (2048, 8192), dtype=torch.int32)
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs4 = torch.randint(0, 100, (125, 5120), dtype=torch.int32)
    
    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4]
    ]


def get_init_inputs():
    # Fixed parameters for integer to float conversion
    dtype = torch.float32  # Target dtype for conversion
    return [dtype]