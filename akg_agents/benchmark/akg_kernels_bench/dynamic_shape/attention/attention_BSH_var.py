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


def get_inputs_dyn_list():
    # Case 1: Small tensor size (1, 1, 128) (aligned)
    query1 = torch.randn(1, 1, 128, dtype=torch.float16)
    key1 = torch.randn(1, 1, 128, dtype=torch.float16)
    value1 = torch.randn(1, 1, 128, dtype=torch.float16)
    
    # Case 2: Small tensor size (1, 1, 256) (aligned)
    query2 = torch.randn(1, 1, 256, dtype=torch.float16)
    key2 = torch.randn(1, 1, 256, dtype=torch.float16)
    value2 = torch.randn(1, 1, 256, dtype=torch.float16)
    
    # Case 3: Small tensor size (4, 4, 512) (aligned)
    query3 = torch.randn(4, 4, 512, dtype=torch.float16)
    key3 = torch.randn(4, 4, 512, dtype=torch.float16)
    value3 = torch.randn(4, 4, 512, dtype=torch.float16)
    
    # Case 4: Medium tensor size (4, 4, 512) (diff, aligned)
    query4 = torch.randn(4, 4, 512, dtype=torch.float16)
    key4 = torch.randn(4, 4, 256, dtype=torch.float16)
    value4 = torch.randn(4, 4, 256, dtype=torch.float16)
    
    # Case 5: Medium tensor size (8, 8, 512) (diff, aligned)
    query5 = torch.randn(8, 8, 512, dtype=torch.float16)
    key5 = torch.randn(8, 8, 256, dtype=torch.float16)
    value5 = torch.randn(8, 8, 256, dtype=torch.float16)
    
    # Case 6: Extreme tensor size (32, 8, 512) (aligned)
    query6 = torch.randn(32, 8, 512, dtype=torch.float16)
    key6 = torch.randn(32, 8, 512, dtype=torch.float16)
    value6 = torch.randn(32, 8, 512, dtype=torch.float16)
    
    return [
        [query1, key1, value1],
        [query2, key2, value2],
        [query3, key3, value3],
        [query4, key4, value4],
        [query5, key5, value5],
        [query6, key6, value6]
    ]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = False  # Not using Grouped-Query Attention
    head_dim = 128
    return [dropout_p, is_causal, enable_gqa, head_dim]