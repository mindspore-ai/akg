import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

    def forward(self, query, key, value, attn_mask=None):        
        # torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        # Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed,
        # and applying dropout if a probability greater than 0.0 is specified.
        # This is a flash attention implementation with causal masking.
        # This is the core computation in transformer models for relating different positions of the input sequence.
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs_dyn_list():
    # Grouped-Query Attention variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small tensor size (4, 4, 64, 16) (non-aligned)
    # 4 query heads, 2 key/value heads (2 groups, 2 heads per group)
    query1 = torch.randn(4, 4, 64, 16, dtype=torch.bfloat16)
    key1 = torch.randn(4, 2, 64, 16, dtype=torch.bfloat16)
    value1 = torch.randn(4, 2, 64, 16, dtype=torch.bfloat16)
    
    # Case 2: Small tensor size (4, 4, 64, 16) (aligned)
    # 4 query heads, 2 key/value heads (2 groups, 2 heads per group)
    query2 = torch.randn(4, 4, 64, 16, dtype=torch.bfloat16)
    key2 = torch.randn(4, 2, 64, 16, dtype=torch.bfloat16)
    value2 = torch.randn(4, 2, 64, 16, dtype=torch.bfloat16)
    
    # Case 3: Medium tensor size (8, 8, 128, 32) (non-aligned)
    # 8 query heads, 4 key/value heads (4 groups, 2 heads per group)
    query3 = torch.randn(8, 8, 128, 32, dtype=torch.bfloat16)
    key3 = torch.randn(8, 4, 128, 32, dtype=torch.bfloat16)
    value3 = torch.randn(8, 4, 128, 32, dtype=torch.bfloat16)
    
    # Case 4: Medium tensor size (8, 8, 128, 32) (aligned)
    # 8 query heads, 4 key/value heads (4 groups, 2 heads per group)
    query4 = torch.randn(8, 8, 128, 32, dtype=torch.bfloat16)
    key4 = torch.randn(8, 4, 128, 32, dtype=torch.bfloat16)
    value4 = torch.randn(8, 4, 128, 32, dtype=torch.bfloat16)
    
    # Case 5: Large tensor size (16, 16, 256, 64) (non-aligned)
    # 16 query heads, 8 key/value heads (8 groups, 2 heads per group)
    query5 = torch.randn(16, 16, 256, 64, dtype=torch.bfloat16)
    key5 = torch.randn(16, 8, 256, 64, dtype=torch.bfloat16)
    value5 = torch.randn(16, 8, 256, 64, dtype=torch.bfloat16)
    
    return [
        [query1, key1, value1],
        [query2, key2, value2],
        [query3, key3, value3],
        [query4, key4, value4],
        [query5, key5, value5]
    ]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = True  # Using Grouped-Query Attention
    return [dropout_p, is_causal, enable_gqa]