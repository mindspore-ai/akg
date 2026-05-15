import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_groups=32):
        super(Model, self).__init__()
        self.num_groups = num_groups

    def forward(self, input_tensor):
        # GroupNormSwish operation
        
        # Apply GroupNorm
        group_norm = torch.nn.functional.group_norm(input_tensor, self.num_groups)
        
        # Apply Swish activation (sigmoid(x) * x)
        result = torch.sigmoid(group_norm) * group_norm
        
        return result

def get_inputs_dyn_list():
    # GroupNormSwish variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small tensor size (4, 512, 28, 28) (aligned channels)
    input_tensor1 = torch.randn(4, 512, 28, 28, dtype=torch.float32)
    
    # Case 2: Medium tensor size (8, 1024, 32, 32) (aligned)
    input_tensor2 = torch.randn(8, 1024, 32, 32, dtype=torch.float32)
    
    # Case 3: Large tensor size (16, 1536, 56, 56) (non-aligned channels)
    input_tensor3 = torch.randn(16, 1536, 56, 56, dtype=torch.float32)
    
    # Case 4: Very large tensor size (32, 2048, 64, 64) (aligned)
    input_tensor4 = torch.randn(32, 2048, 64, 64, dtype=torch.float32)
    
    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]

def get_init_inputs():
    # Parameters for GroupNormSwish operation
    num_groups = 32  # Number of groups for GroupNorm
    return [num_groups]
