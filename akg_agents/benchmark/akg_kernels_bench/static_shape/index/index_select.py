import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, indices):
        # IndexSelect operation
        # This operation is commonly used in neural networks for:
        # - Selecting specific elements from a tensor along a dimension
        # - Used in embedding lookups and gather operations
        # - Implementing sparse operations
        
        # Perform index selection
        result = torch.index_select(var, self.dim, indices)
        
        return result

def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    var = torch.randn(1024, 4096, dtype=torch.float32)
    indices = torch.tensor([0, 2, 4], dtype=torch.int64)
    return [var, indices]

def get_init_inputs():
    # Parameters for IndexSelect operation
    dim = 1  # Dimension along which to select indices
    return [dim]