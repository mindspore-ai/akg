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

def get_inputs():
    # Batch size: 32
    # Channels: 2688
    # Feature map size: 56 x 56
    input_tensor = torch.randn(32, 2688, 56, 56, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    # Parameters for GroupNormSwish operation
    num_groups = 32  # Number of groups for GroupNorm
    return [num_groups]