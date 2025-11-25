import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, scale_value=0.088388, keep_prob=1.0, pre_tokens=2147483647, next_tokens=2147483647, 
                 inner_precise=0, sparse_mod=0, layout="SBH"):
        super(Model, self).__init__()
        self.scale_value = scale_value
        self.keep_prob = keep_prob
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.inner_precise = inner_precise
        self.sparse_mod = sparse_mod
        self.layout = layout

    def forward(self, var, var_scale=None, indices=None):
        # Using var_scale as key, indices as value
        query = var
        key = var_scale
        value = indices
        
        # Flash Attention Score computation
        # This operation is commonly used in neural networks for:
        # - Efficiently computing scaled dot-product attention in transformer models
        # - Reducing memory usage during attention computation
        # - Accelerating training and inference in large transformer models
        
        # Apply the scaling factor
        scaled_query = query * self.scale_value
        
        # Compute attention scores (Q @ K^T)
        # Note: The actual flash attention implementation is much more complex and optimized
        # This is a simplified version for demonstration purposes
        attention_scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        
        # Apply softmax to get attention weights (using fp32 for intermediate computation)
        attention_scores_fp32 = attention_scores.to(torch.float32)
        attention_weights_fp32 = torch.nn.functional.softmax(attention_scores_fp32, dim=-1)
        attention_weights = attention_weights_fp32.to(query.dtype)
        
        # Apply dropout if keep_prob < 1.0
        if self.keep_prob < 1.0:
            attention_weights = torch.nn.functional.dropout(attention_weights, p=1-self.keep_prob)
        
        # Compute the final attention output (weights @ V)
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output

def get_inputs_dyn_list():
    # Flash attention score variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small tensor size (15, 1, 128) (non-aligned)
    query1 = torch.randn(15, 1, 128, dtype=torch.bfloat16)
    key1 = torch.randn(15, 1, 128, dtype=torch.bfloat16)
    value1 = torch.randn(15, 1, 128, dtype=torch.bfloat16)
    
    # Case 2: Small tensor size (31, 1, 128) (non-aligned)
    query2 = torch.randn(31, 1, 128, dtype=torch.bfloat16)
    key2 = torch.randn(31, 1, 128, dtype=torch.bfloat16)
    value2 = torch.randn(31, 1, 128, dtype=torch.bfloat16)
    
    # Case 3: Small tensor size (32, 1, 128) (aligned)
    query3 = torch.randn(32, 1, 128, dtype=torch.bfloat16)
    key3 = torch.randn(32, 1, 128, dtype=torch.bfloat16)
    value3 = torch.randn(32, 1, 128, dtype=torch.bfloat16)
    
    # Case 4: Medium tensor size (63, 1, 128) (non-aligned)
    query4 = torch.randn(63, 1, 128, dtype=torch.bfloat16)
    key4 = torch.randn(63, 1, 128, dtype=torch.bfloat16)
    value4 = torch.randn(63, 1, 128, dtype=torch.bfloat16)
    
    # Case 5: Medium tensor size (64, 1, 128) (aligned)
    query5 = torch.randn(64, 1, 128, dtype=torch.bfloat16)
    key5 = torch.randn(64, 1, 128, dtype=torch.bfloat16)
    value5 = torch.randn(64, 1, 128, dtype=torch.bfloat16)
    
    # Case 6: Large tensor size (127, 1, 128) (non-aligned)
    query6 = torch.randn(127, 1, 128, dtype=torch.bfloat16)
    key6 = torch.randn(127, 1, 128, dtype=torch.bfloat16)
    value6 = torch.randn(127, 1, 128, dtype=torch.bfloat16)
    
    # Case 7: Large tensor size (128, 1, 128) (aligned)
    query7 = torch.randn(128, 1, 128, dtype=torch.bfloat16)
    key7 = torch.randn(128, 1, 128, dtype=torch.bfloat16)
    value7 = torch.randn(128, 1, 128, dtype=torch.bfloat16)
    
    # Case 8: Very large tensor size (255, 1, 128) (non-aligned)
    query8 = torch.randn(255, 1, 128, dtype=torch.bfloat16)
    key8 = torch.randn(255, 1, 128, dtype=torch.bfloat16)
    value8 = torch.randn(255, 1, 128, dtype=torch.bfloat16)
    
    # Case 9: Very large tensor size (256, 1, 128) (aligned)
    query9 = torch.randn(256, 1, 128, dtype=torch.bfloat16)
    key9 = torch.randn(256, 1, 128, dtype=torch.bfloat16)
    value9 = torch.randn(256, 1, 128, dtype=torch.bfloat16)
    
    # Case 10: Extreme tensor size (511, 1, 128) (non-aligned)
    query10 = torch.randn(511, 1, 128, dtype=torch.bfloat16)
    key10 = torch.randn(511, 1, 128, dtype=torch.bfloat16)
    value10 = torch.randn(511, 1, 128, dtype=torch.bfloat16)
    
    return [
        [query1, key1, value1],
        [query2, key2, value2],
        [query3, key3, value3],
        [query4, key4, value4],
        [query5, key5, value5],
        [query6, key6, value6],
        [query7, key7, value7],
        [query8, key8, value8],
        [query9, key9, value9],
        [query10, key10, value10]
    ]

def get_init_inputs():
    # Fixed parameters for flash attention score
    scale_value = 0.088388      # Scaling factor for query
    keep_prob = 1.0             # Dropout probability (1.0 means no dropout)
    pre_tokens = 2147483647     # Number of tokens to process before
    next_tokens = 2147483647    # Number of tokens to process after
    inner_precise = 0           # Precision flag
    sparse_mod = 0              # Sparsity mode
    layout = "SBH"              # Tensor layout (Sequence, Batch, Hidden)
    return [scale_value, keep_prob, pre_tokens, next_tokens, inner_precise, sparse_mod, layout]