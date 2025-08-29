import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Cumulative Sum operation that computes the cumulative sum of elements along a dimension.
    This operation is commonly used in neural networks for:
    - Computing cumulative distributions
    - Used in some attention mechanisms and sequence processing
    - Computing prefix sums in algorithms
    
    Formula: output[i] = sum(input[0:i+1])
    """
    def __init__(self, dim=-1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # Cumulative sum operation using var as input
        # var_scale, indices, updates, and smooth_scales are not used in this operation

        result = torch.cumsum(var, dim=self.dim)
        return result


def get_inputs_dyn_list():
    # Cumulative sum along dimension 1 variation cases with both aligned and non-aligned shapes

    # Case 1: Large tensor size 512x512 (aligned)
    inputs1 = torch.randn(512, 512, dtype=torch.float32)

    # Case 2: Very large tensor size 1023x1023 (non-aligned)
    inputs2 = torch.randn(1023, 1023, dtype=torch.float32)

    # Case 3: Very large tensor size 1024x1024 (aligned)
    inputs3 = torch.randn(1024, 1024, dtype=torch.float32)

    # Case 4: Extreme tensor size 4096x4096 (aligned)
    inputs4 = torch.randn(4096, 4096, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
    ]

def get_init_inputs():
    # Fixed parameters for cumulative sum along dimension 1
    dim = 1  # Dimension along which to compute cumulative sum
    return [dim]
