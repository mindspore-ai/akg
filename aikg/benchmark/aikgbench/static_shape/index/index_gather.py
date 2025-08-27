import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, index_tensor):
        # Gather operation
        # This operation is commonly used in neural networks for:
        # - Collecting values from specific indices
        # - Used in attention mechanisms and embedding lookups
        # - Implementing advanced indexing operations
        
        # Perform gather operation
        result = torch.gather(input_tensor, self.dim, index_tensor)
        
        return result

def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Index tensor shape: [1024, 2048]
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    index_tensor = torch.randint(0, 4096, (1024, 2048), dtype=torch.int64)
    return [input_tensor, index_tensor]

def get_init_inputs():
    # Parameters for Gather operation
    dim = 1  # Dimension along which to gather values
    return [dim]