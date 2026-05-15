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

def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    input_tensor1 = torch.randn(15, 1344, dtype=torch.float32)
    index_tensor1 = torch.randint(0, 1344, (15, 672), dtype=torch.int64)
    
    # Case 2: Small batch, large hidden (aligned batch)
    input_tensor2 = torch.randn(16, 4096, dtype=torch.float32)
    index_tensor2 = torch.randint(0, 4096, (16, 2048), dtype=torch.int64)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    input_tensor3 = torch.randn(127, 2688, dtype=torch.float32)
    index_tensor3 = torch.randint(0, 2688, (127, 1344), dtype=torch.int64)
    
    # Case 4: Large batch, large hidden (aligned batch)
    input_tensor4 = torch.randn(512, 5120, dtype=torch.float32)
    index_tensor4 = torch.randint(0, 5120, (512, 2560), dtype=torch.int64)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    input_tensor5 = torch.randn(1023, 8192, dtype=torch.float32)
    index_tensor5 = torch.randint(0, 8192, (1023, 4096), dtype=torch.int64)
    
    return [
        [input_tensor1, index_tensor1],
        [input_tensor2, index_tensor2],
        [input_tensor3, index_tensor3],
        [input_tensor4, index_tensor4],
        [input_tensor5, index_tensor5]
    ]

def get_init_inputs():
    # Parameters for Gather operation
    dim = 1  # Dimension along which to gather values
    return [dim]