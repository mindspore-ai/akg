import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, reduction="sum"):
        super(Model, self).__init__()
        self.reduction = reduction

    def forward(self, input_tensor, index_tensor, src_tensor):
        # ScatterAddWithSorted operation
        # This operation is commonly used in neural networks for:
        # - Accumulating values at specific indices
        # - Used in sparse operations and graph neural networks
        # - Implementing message passing in graphs
        
        # Perform scatter add operation
        result = torch.scatter_add(input_tensor, 0, index_tensor, src_tensor)
        
        return result

def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Index tensor shape: [512, 4096]
    # Source tensor shape: [512, 4096]
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    index_tensor = torch.randint(0, 1024, (512, 4096), dtype=torch.int64)
    src_tensor = torch.randn(512, 4096, dtype=torch.float32)
    return [input_tensor, index_tensor, src_tensor]

def get_init_inputs():
    # Parameters for ScatterAddWithSorted operation
    reduction = "sum"  # Reduction method
    return [reduction]