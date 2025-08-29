import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.sum(input, dim, keepdim=False, dtype=None)
        # Returns the sum of each row of the input tensor in the given dimension dim.
        # This is commonly used in neural networks for:
        # - Computing loss over a batch or aggregating features
        # - Reducing tensor dimensions
        return torch.sum(var, self.dim)


def get_inputs_dyn_list():
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
    # Shape (125, 3072) represents a batch of 125 samples with 3072 features each
    inputs4 = torch.randn(125, 3072, dtype=torch.float32)
    
    return [
        [inputs1],  # Case 1 inputs
        [inputs2],  # Case 2 inputs
        [inputs3],  # Case 3 inputs
        [inputs4]   # Case 4 inputs
    ]


def get_init_inputs():
    # Specific dim value for reduction
    # Extract params  # Reduce along second dimension (features dimension)
    return []
