import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Top-K operation that selects the top-K elements along a specified dimension.
    This operation is commonly used in neural networks for:
    - Selecting top predictions in classification tasks
    - Used in beam search and other search algorithms
    - Core component in sparse attention mechanisms
    
    Formula: Returns the K largest elements and their indices
    """
    def __init__(self, k=10):
        super(Model, self).__init__()
        self.k = k

    def forward(self, logits):
        # Top-K operation using logits as input
        # Apply top-k operation
        values, indices = torch.topk(logits, self.k, dim=-1)
        return values, indices

def get_inputs_dyn_list():
    # Case 1: Small batch, small vocab (non-aligned batch)
    logits1 = torch.randn(15, 1024, dtype=torch.float16)
    
    # Case 2: Small batch, medium vocab (aligned batch)
    logits2 = torch.randn(16, 8192, dtype=torch.float16)
    
    # Case 3: Medium batch, large vocab (non-aligned batch)
    logits3 = torch.randn(63, 32000, dtype=torch.float16)
    
    # Case 4: Large batch, very large vocab (aligned batch)
    logits4 = torch.randn(128, 50257, dtype=torch.float16)
    
    # Case 5: Very large batch, very large vocab (non-aligned batch)
    logits5 = torch.randn(255, 65536, dtype=torch.float16)
    
    return [
        [logits1],
        [logits2],
        [logits3],
        [logits4],
        [logits5]
    ]

def get_init_inputs():
    # Parameters for Top-K operation
    k = 10
    return [k]