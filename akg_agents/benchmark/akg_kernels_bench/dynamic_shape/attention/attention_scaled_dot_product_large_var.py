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
    # Scaled dot product attention with large tensors variation cases with both aligned and non-aligned shapes
    
    # Case 1: Medium tensor size (4, 4, 63, 63) (non-aligned)
    query1 = torch.randn(4, 4, 63, 63, dtype=torch.bfloat16)
    key1 = torch.randn(4, 4, 63, 63, dtype=torch.bfloat16)
    value1 = torch.randn(4, 4, 63, 63, dtype=torch.bfloat16)
    
    # Case 2: Medium tensor size (4, 4, 64, 64) (aligned)
    query2 = torch.randn(4, 4, 64, 64, dtype=torch.bfloat16)
    key2 = torch.randn(4, 4, 64, 64, dtype=torch.bfloat16)
    value2 = torch.randn(4, 4, 64, 64, dtype=torch.bfloat16)
    
    # Case 3: Large tensor size (8, 8, 127, 127) (non-aligned)
    query3 = torch.randn(8, 8, 127, 127, dtype=torch.bfloat16)
    key3 = torch.randn(8, 8, 127, 127, dtype=torch.bfloat16)
    value3 = torch.randn(8, 8, 127, 127, dtype=torch.bfloat16)
    
    # Case 4: Large tensor size (8, 8, 128, 128) (aligned)
    query4 = torch.randn(8, 8, 128, 128, dtype=torch.bfloat16)
    key4 = torch.randn(8, 8, 128, 128, dtype=torch.bfloat16)
    value4 = torch.randn(8, 8, 128, 128, dtype=torch.bfloat16)
    
    # Case 5: Very large tensor size (16, 16, 255, 255) (non-aligned)
    query5 = torch.randn(16, 16, 255, 255, dtype=torch.bfloat16)
    key5 = torch.randn(16, 16, 255, 255, dtype=torch.bfloat16)
    value5 = torch.randn(16, 16, 255, 255, dtype=torch.bfloat16)
    
    # Case 6: Very large tensor size (16, 16, 256, 256) (aligned)
    query6 = torch.randn(16, 16, 256, 256, dtype=torch.bfloat16)
    key6 = torch.randn(16, 16, 256, 256, dtype=torch.bfloat16)
    value6 = torch.randn(16, 16, 256, 256, dtype=torch.bfloat16)
    
    # Case 7: Very large tensor size (32, 32, 511, 128) (non-aligned)
    query7 = torch.randn(32, 32, 511, 128, dtype=torch.bfloat16)
    key7 = torch.randn(32, 32, 511, 128, dtype=torch.bfloat16)
    value7 = torch.randn(32, 32, 511, 128, dtype=torch.bfloat16)
    
    # Case 8: Very large tensor size (32, 32, 512, 128) (aligned)
    query8 = torch.randn(32, 32, 512, 128, dtype=torch.bfloat16)
    key8 = torch.randn(32, 32, 512, 128, dtype=torch.bfloat16)
    value8 = torch.randn(32, 32, 512, 128, dtype=torch.bfloat16)
    
    # Case 9: Extreme tensor size (64, 32, 1023, 128) (non-aligned)
    query9 = torch.randn(64, 32, 1023, 128, dtype=torch.bfloat16)
    key9 = torch.randn(64, 32, 1023, 128, dtype=torch.bfloat16)
    value9 = torch.randn(64, 32, 1023, 128, dtype=torch.bfloat16)
    
    # Case 10: Extreme tensor size (64, 32, 1024, 128) (aligned)
    query10 = torch.randn(64, 32, 1024, 128, dtype=torch.bfloat16)
    key10 = torch.randn(64, 32, 1024, 128, dtype=torch.bfloat16)
    value10 = torch.randn(64, 32, 1024, 128, dtype=torch.bfloat16)
    
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
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = False  # Not using Grouped-Query Attention
    return [dropout_p, is_causal, enable_gqa]