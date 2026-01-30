import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, top_k=2):
        super(Model, self).__init__()
        self.top_k = top_k

    def forward(self, gating_logits):
        # MoE Gating Top-K Softmax
        # This operation is commonly used in neural networks for:
        # - Implementing Mixture of Experts (MoE) routing in large models
        # - Selecting top-K experts for each token based on learned gates
        # - Enabling model scaling with conditional computation
        
        # Apply softmax to get gating probabilities
        gating_probs = torch.nn.functional.softmax(gating_logits, dim=-1)
        
        # Select top-K experts for each token
        top_k_probs, top_k_indices = torch.topk(gating_probs, self.top_k, dim=-1)
        
        # Normalize the top-K probabilities
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices

def get_inputs_dyn_list():
    # Case 1: Small batch, small experts (non-aligned batch)
    gating_logits1 = torch.randn(15, 8, dtype=torch.float32)
    
    # Case 2: Small batch, medium experts (aligned batch)
    gating_logits2 = torch.randn(16, 16, dtype=torch.float32)
    
    # Case 3: Medium batch, medium experts (non-aligned batch)
    gating_logits3 = torch.randn(127, 32, dtype=torch.float32)
    
    # Case 4: Large batch, large experts (aligned batch)
    gating_logits4 = torch.randn(512, 64, dtype=torch.float32)
    
    # Case 5: Very large batch, very large experts (non-aligned batch)
    gating_logits5 = torch.randn(1023, 128, dtype=torch.float32)
    
    return [
        [gating_logits1],
        [gating_logits2],
        [gating_logits3],
        [gating_logits4],
        [gating_logits5]
    ]

def get_init_inputs():
    # Parameters for MoE Gating Top-K Softmax
    top_k = 2  # Number of top experts to select
    return [top_k]