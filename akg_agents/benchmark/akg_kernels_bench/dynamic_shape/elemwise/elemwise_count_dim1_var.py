import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, value):
        # torch.eq(input, other, *, out=None)
        # Computes element-wise equality.
        # Then we count the number of True values along the specified dimension.
        # This operation is commonly used in neural networks for:
        # - Counting elements that meet specific criteria
        # - Computing statistics on boolean masks
        # - Implementing certain counting mechanisms
        condition = (input_tensor == value)
        return torch.sum(condition, dim=1)


def get_inputs_dyn_list():
    # Count along dimension 0 variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inputs1 = torch.randn(256, 1024, dtype=torch.float32)
    cmp_val1 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inputs2 = torch.randn(1024, 4096, dtype=torch.float32)
    cmp_val2 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inputs3 = torch.randn(2048, 8192, dtype=torch.float32)
    cmp_val3 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs4 = torch.randn(125, 5120, dtype=torch.float32)
    cmp_val4 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    return [
        [inputs1, cmp_val1],
        [inputs2, cmp_val2],
        [inputs3, cmp_val3],
        [inputs4, cmp_val4]
    ]


def get_init_inputs():
    return []