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

def get_inputs_dyn_list():
    # Small case
    tensor1_1 = torch.randn(16, 512, dtype=torch.float32)
    tensor1_2 = torch.randn(16, 512, dtype=torch.float32)
    # Middle case
    tensor2_1 = torch.randn(32, 2048, dtype=torch.float32)
    tensor2_2 = torch.randn(32, 2048, dtype=torch.float32)
    # Large case
    tensor3_1 = torch.randn(64, 4096, dtype=torch.float32)
    tensor3_2 = torch.randn(64, 4096, dtype=torch.float32)
    # Noaligned case
    tensor4_1 = torch.randn(65, 2688, dtype=torch.float32)
    tensor4_2 = torch.randn(65, 2688, dtype=torch.float32)
    return [
        [tensor1_1, tensor1_2],
        [tensor2_1, tensor2_2],
        [tensor3_1, tensor3_2],
        [tensor4_1, tensor4_2]
    ]

def get_init_inputs():
    # Parameters for AddLayerNorm operation
    dim = 1
    return [dim]