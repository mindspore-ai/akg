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


def get_inputs_dyn_list():
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inputs1 = torch.randn(256, 1024, dtype=torch.float64)
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inputs2 = torch.randn(1024, 4096, dtype=torch.float64)
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inputs3 = torch.randn(2048, 8192, dtype=torch.float64)
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs4 = torch.randn(125, 5120, dtype=torch.float64)
    
    return [
        [inputs1],  # Case 1 inputs
        [inputs2],  # Case 2 inputs
        [inputs3],  # Case 3 inputs
        [inputs4]   # Case 4 inputs
    ]


def get_init_inputs():
    # Parameters for cast
    dtype = torch.float32  # Target dtype
    return [dtype]