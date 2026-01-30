import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, value):
        # torch.eq(input, other, *, out=None)
        # Computes element-wise equality.
        # Then we count the number of True values along the specified dimension.
        # This operation is commonly used in neural networks for:
        # - Counting elements that meet specific criteria
        # - Computing statistics on boolean masks
        # - Implementing certain counting mechanisms
        condition = (input_tensor == value)
        return torch.sum(condition, dim=self.dim)


def get_inputs_dyn_list():
    # Count along dimension 1 variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    inputs1 = torch.randn(256, 4096, dtype=torch.float32)
    cmp_val1 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs2 = torch.randn(125, 5120, dtype=torch.float32)
    cmp_val2 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    inputs3 = torch.randn(512, 6144, dtype=torch.float32)
    cmp_val3 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    inputs4 = torch.randn(1024, 8192, dtype=torch.float32)
    cmp_val4 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    
    return [
        [inputs1, cmp_val1],
        [inputs2, cmp_val2],
        [inputs3, cmp_val3],
        [inputs4, cmp_val4]
    ]


def get_init_inputs():
    # Fixed parameters for count along dimension 1
    dim = 1  # Count along second dimension (features dimension)
    return [dim]