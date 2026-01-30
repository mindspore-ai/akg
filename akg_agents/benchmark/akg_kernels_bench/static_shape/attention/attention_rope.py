import torch
import torch.nn as nn
import math


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False, max_seq_len=2048):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.max_seq_len = max_seq_len
        self.enable_gqa = enable_gqa

    def forward(self, query, key, value, attn_mask=None):
        # Apply rotary position embeddings to query and key
        query_rot = self._apply_rotary_embeddings(query)
        key_rot = self._apply_rotary_embeddings(key)
        
        # torch.nn.functional.scaled_dot_product_attention
        # Computes scaled dot product attention with rotary position embeddings
        return torch.nn.functional.scaled_dot_product_attention(
            query_rot, key_rot, value, attn_mask=attn_mask, 
            dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )
    
    def _apply_rotary_embeddings(self, x):
        """
        Apply rotary position embeddings to the input tensor.
        """
        batch, num_heads, seq_len, head_dim = x.shape
        
        # Create position indices
        pos = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        
        # Create frequency bands
        dim = head_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(dim, dtype=x.dtype, device=x.device) / dim)
        freqs = pos * freqs.unsqueeze(0)
        
        # Apply rotation
        cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, dim)
        sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, dim)
        
        # Reshape x to separate even and odd dimensions
        x_even = x[..., ::2]  # Even dimensions
        x_odd = x[..., 1::2]  # Odd dimensions
        
        # Apply rotation matrix
        x_rot = torch.cat([
            x_even * cos_freqs - x_odd * sin_freqs,
            x_odd * cos_freqs + x_even * sin_freqs
        ], dim=-1)
        
        return x_rot


def get_inputs():
    # Using shapes that are representative of large model computations in transformer models
    # Shape (32, 8, 1024, 64) represents:
    # - 32 batches
    # - 8 attention heads
    # - 1024 sequence length
    # - 64 head dimension
    batch, num_heads, seq_len, head_dim = 32, 8, 1024, 64
    shape = (batch, num_heads, seq_len, head_dim)
    
    query = torch.randn(shape, dtype=torch.bfloat16)
    key = torch.randn(shape, dtype=torch.bfloat16)
    value = torch.randn(shape, dtype=torch.bfloat16)
    return [query, key, value]


def get_init_inputs():
    # Parameters for rotary position embedding attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = True  # Using Grouped-Query Attention
    max_seq_len = 2048  # Maximum sequence length
    return [dropout_p, is_causal, enable_gqa, max_seq_len]