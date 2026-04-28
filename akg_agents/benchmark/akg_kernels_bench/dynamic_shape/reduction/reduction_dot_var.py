import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, inp1, var_scale=None, indices=None, updates=None, smooth_scales=None):
        # Using var_scale as the second operand (inp2)
        inp2 = var_scale
        
        # torch.dot(input, other)
        # Computes the dot product of two 1D tensors.
        # The dot product is calculated as the sum of the element-wise products.
        # This operation is commonly used in neural networks for:
        # - Computing similarity between vectors
        # - Implementing attention mechanisms
        # - Calculating projections in certain layers
        return torch.dot(inp1, inp2)


def get_inputs_dyn_list():
    # Dot product variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small vector size 128 (aligned)
    inp1_1 = torch.randn(128, dtype=torch.float32)
    inp2_1 = torch.randn(128, dtype=torch.float32)
    
    # Case 2: Large vector size 511 (non-aligned)
    inp1_2 = torch.randn(511, dtype=torch.float32)
    inp2_2 = torch.randn(511, dtype=torch.float32)
    
    # Case 3: Very large vector size 1024 (aligned)
    inp1_3 = torch.randn(1024, dtype=torch.float32)
    inp2_3 = torch.randn(1024, dtype=torch.float32)
    
    # Case 4: Extreme vector size 4096 (aligned)
    inp1_4 = torch.randn(4096, dtype=torch.float32)
    inp2_4 = torch.randn(4096, dtype=torch.float32)
    
    return [
        [inp1_1, inp2_1],
        [inp1_2, inp2_2],
        [inp1_3, inp2_3],
        [inp1_4, inp2_4],
    ]


def get_init_inputs():
    # No parameters needed for dot product
    return []