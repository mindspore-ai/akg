import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.bitwise_not(input, *, out=None)
        # Computes the element-wise bitwise NOT of the given input tensor.
        # This operation is commonly used in neural networks for:
        # - Implementing bit manipulation operations
        # - Creating bit masks
        # - Low-level data processing
        return torch.bitwise_not(input_tensor)


def get_inputs_dyn_list():
    # Element-wise bitwise NOT variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    inputs1 = torch.randint(0, 256, (256, 4096), dtype=torch.int32)
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs2 = torch.randint(0, 256, (125, 5120), dtype=torch.int32)
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    inputs3 = torch.randint(0, 256, (512, 6144), dtype=torch.int32)
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    inputs4 = torch.randint(0, 256, (1024, 8192), dtype=torch.int32)
    
    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4]
    ]


def get_init_inputs():
    # No parameters needed for bitwise_not
    return []