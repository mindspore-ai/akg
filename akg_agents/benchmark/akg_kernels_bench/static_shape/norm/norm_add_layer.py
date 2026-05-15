import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, tensor1, tensor2):
        # AddLayerNorm operation
        # This operation is commonly used in neural networks for:
        # - Combining residual connections with layer normalization
        # - Used in transformer architectures
        # - Improving training stability and convergence
        
        # Perform addition
        added = tensor1 + tensor2
        
        # Apply layer normalization
        result = torch.nn.functional.layer_norm(added, added.shape[1:])
        
        return result

def get_inputs():
    # Batch size: 32
    # Hidden dimension: 4096
    tensor1 = torch.randn(32, 4096, dtype=torch.float32)
    tensor2 = torch.randn(32, 4096, dtype=torch.float32)
    return [tensor1, tensor2]

def get_init_inputs():
    # Parameters for AddLayerNorm operation
    dim = 1
    return [dim]