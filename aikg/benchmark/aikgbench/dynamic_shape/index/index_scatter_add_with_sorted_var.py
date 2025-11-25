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

def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    input_tensor1 = torch.randn(15, 1344, dtype=torch.float32)
    index_tensor1 = torch.randint(0, 15, (7, 1344), dtype=torch.int64)
    src_tensor1 = torch.randn(7, 1344, dtype=torch.float32)
    
    # Case 2: Small batch, large hidden (aligned batch)
    input_tensor2 = torch.randn(16, 4096, dtype=torch.float32)
    index_tensor2 = torch.randint(0, 16, (8, 4096), dtype=torch.int64)
    src_tensor2 = torch.randn(8, 4096, dtype=torch.float32)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    input_tensor3 = torch.randn(127, 2688, dtype=torch.float32)
    index_tensor3 = torch.randint(0, 127, (63, 2688), dtype=torch.int64)
    src_tensor3 = torch.randn(63, 2688, dtype=torch.float32)
    
    # Case 4: Large batch, large hidden (aligned batch)
    input_tensor4 = torch.randn(512, 5120, dtype=torch.float32)
    index_tensor4 = torch.randint(0, 512, (256, 5120), dtype=torch.int64)
    src_tensor4 = torch.randn(256, 5120, dtype=torch.float32)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    input_tensor5 = torch.randn(1023, 8192, dtype=torch.float32)
    index_tensor5 = torch.randint(0, 1023, (511, 8192), dtype=torch.int64)
    src_tensor5 = torch.randn(511, 8192, dtype=torch.float32)
    
    return [
        [input_tensor1, index_tensor1, src_tensor1],
        [input_tensor2, index_tensor2, src_tensor2],
        [input_tensor3, index_tensor3, src_tensor3],
        [input_tensor4, index_tensor4, src_tensor4],
        [input_tensor5, index_tensor5, src_tensor5]
    ]

def get_init_inputs():
    # Parameters for ScatterAddWithSorted operation
    reduction = "sum"  # Reduction method
    return [reduction]