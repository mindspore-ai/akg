import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_groups):
        super(Model, self).__init__()
        self.num_groups = num_groups

    def forward(self, inp):
        # torch.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-05)
        # Applies Group Normalization over a mini-batch of inputs.
        
        # Create weight and bias tensors with the correct shape
        # The shape should match the channel dimension (inp.shape[1])
        weight = torch.ones(inp.shape[1], dtype=inp.dtype, device=inp.device)
        bias = torch.zeros(inp.shape[1], dtype=inp.dtype, device=inp.device)
        
        return torch.nn.functional.group_norm(inp, self.num_groups, weight, bias)

def get_inputs_dyn_list():
    # Group normalization variation cases with both aligned and non-aligned shapes
    
    # Case 1: Medium tensor size (15, 15, 1344) (non-aligned)
    inp1 = torch.randn(15, 15, 1344, dtype=torch.float32)
    
    # Case 2: Very large tensor size (16, 16, 4096) (aligned)
    inp2 = torch.randn(16, 16, 4096, dtype=torch.float32)
    
    # Case 3: Extreme tensor size (127, 127, 4096) (non-aligned)
    inp3 = torch.randn(127, 127, 4096, dtype=torch.float32)
    
    # Case 4: Extreme tensor size (128, 128, 8192) (aligned)
    inp4 = torch.randn(128, 128, 8192, dtype=torch.float32)
    
    return [
        [inp1],
        [inp2],
        [inp3],
        [inp4],
    ]

def get_init_inputs():
    # Parameters for group_norm
    num_groups = 8  # Using a fixed number of groups for all cases
    return [num_groups]