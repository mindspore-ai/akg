import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False, head_dim=128):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa
        self.head_dim = head_dim

    def forward(self, query, key, value, attn_mask=None):        
        # torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        # Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed,
        # and applying dropout if a probability greater than 0.0 is specified.
        # This is a flash attention implementation with causal masking.
        # This is the core computation in transformer models for relating different positions of the input sequence.
        q_heads = query.shape[-1] // self.head_dim
        kv_heads = key.shape[-1] // self.head_dim
        
        query = query.view(*query.shape[:-1], q_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)  # Transpose to [batch, heads, seq_len, head_dim] for scaled_dot_product_attention
        
        key = key.view(*key.shape[:-1], kv_heads, self.head_dim)
        key = key.permute(0, 2, 1, 3)  # Transpose to [batch, heads, seq_len, head_dim] for scaled_dot_product_attention
        value = value.view(*value.shape[:-1], kv_heads, self.head_dim)
        value = value.permute(0, 2, 1, 3)  # Transpose to [batch, heads, seq_len, head_dim] for scaled_dot_product_attention

        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs():
    # Using shapes that are representative of large model computations in transformer models
    # Shape (32, 1024, 512) represents:
    # - 32 batches
    # - 1024 sequence length
    # - 512 head dimension * heads
    batch, seq_len, q_head, kv_head = 32, 1024, 512, 256
    q_shape = (batch, seq_len, q_head)
    kv_shape = (batch, seq_len, kv_head)
    
    query = torch.randn(q_shape, dtype=torch.float16)
    key = torch.randn(kv_shape, dtype=torch.float16)
    value = torch.randn(kv_shape, dtype=torch.float16)
    return [query, key, value]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = False  # Not using Grouped-Query Attention
    head_dim = 128
    return [dropout_p, is_causal, enable_gqa, head_dim]