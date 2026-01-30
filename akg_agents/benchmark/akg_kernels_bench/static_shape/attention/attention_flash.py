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
    # Shape (32, 8, 1024, 64) represents:
    # - 32 batches
    # - 8 attention heads
    # - 1024 sequence length
    # - 64 head dimension
    batch, num_heads, seq_len, head_dim = 32, 8, 1024, 64
    shape = (batch, num_heads, seq_len, head_dim)
    
    query = torch.randn(shape, dtype=torch.float16)
    key = torch.randn(shape, dtype=torch.float16)
    value = torch.randn(shape, dtype=torch.float16)
    return [query, key, value]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = False  # Not using Grouped-Query Attention
    return [dropout_p, is_causal, enable_gqa]