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


def get_inputs():
    # Using shapes that are representative of large model computations in transformer models
    # For GQA with 8 query heads and 4 groups:
    # - Query: (32, 8, 1024, 64) - 8 separate query heads
    # - Key/Value: (32, 4, 1024, 64) - 4 shared key/value heads
    batch, num_query_heads, seq_len, head_dim = 32, 8, 1024, 64
    num_key_value_heads = 4  # Number of key/value heads (groups)
    
    query_shape = (batch, num_query_heads, seq_len, head_dim)
    key_value_shape = (batch, num_key_value_heads, seq_len, head_dim)
    
    query = torch.randn(query_shape, dtype=torch.bfloat16)
    key = torch.randn(key_value_shape, dtype=torch.bfloat16)
    value = torch.randn(key_value_shape, dtype=torch.bfloat16)
    return [query, key, value]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = True  # Using Grouped-Query Attention
    return [dropout_p, is_causal, enable_gqa]