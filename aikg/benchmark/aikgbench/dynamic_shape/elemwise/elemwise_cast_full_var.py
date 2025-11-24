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


def get_inputs_dyn_list():
    # Cast to boolean variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inputs1 = torch.randn(256, 1024, dtype=torch.float32)
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inputs2 = torch.randn(1024, 4096, dtype=torch.float32)
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inputs3 = torch.randn(2048, 8192, dtype=torch.float32)
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs4 = torch.randn(125, 5120, dtype=torch.float32)
    
    return [
        [inputs1],  # Case 1 inputs
        [inputs2],  # Case 2 inputs
        [inputs3],  # Case 3 inputs
        [inputs4]   # Case 4 inputs
    ]


def get_init_inputs():
    # Fixed parameters for cast to boolean
    dtype = torch.bool  # Convert to boolean
    return [dtype]