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

def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    var1 = torch.randn(15, 1344, dtype=torch.float32)
    indices1 = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64)
    
    # Case 2: Small batch, large hidden (aligned batch)
    var2 = torch.randn(16, 4096, dtype=torch.float32)
    indices2 = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.int64)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    var3 = torch.randn(127, 2688, dtype=torch.float32)
    indices3 = torch.tensor([0, 10, 20, 30, 40, 50, 60], dtype=torch.int64)
    
    # Case 4: Large batch, large hidden (aligned batch)
    var4 = torch.randn(512, 5120, dtype=torch.float32)
    indices4 = torch.tensor([0, 50, 100, 150, 200, 250], dtype=torch.int64)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    var5 = torch.randn(1023, 8192, dtype=torch.float32)
    indices5 = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=torch.int64)
    
    return [
        [var1, indices1],
        [var2, indices2],
        [var3, indices3],
        [var4, indices4],
        [var5, indices5]
    ]

def get_init_inputs():
    # Parameters for IndexSelect operation
    dim = 1  # Dimension along which to select indices
    return [dim]