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
    """
    Generate input tensors for attention with Rotary Position Embeddings (RoPE).
    
    Tensor shape: (B, H, L, D)
        B = 32   : Batch size (number of independent sequences)
        H = 8    : Number of attention heads (multi-head attention)
        L = 1024 : Sequence length (number of tokens in each sequence)
        D = 64   : Head dimension (embedding size per attention head)
    
    Total model dimension = H * D = 8 * 64 = 512
    
    Note: RoPE is applied to query and key tensors before attention computation,
          encoding relative positional information through rotation in the complex plane.
          This is used in models like LLaMA, PaLM, and GPT-NeoX.
    """
    B, H, L, D = 32, 8, 1024, 64
    shape = (B, H, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    query = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    key = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    value = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for attention with Rotary Position Embeddings.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: False (not using causal masking)
        enable_gqa: False (RoPE is orthogonal to GQA, not enabled by default)
        max_seq_len: 2048 (maximum sequence length for position embeddings)
    
    Note: RoPE (Rotary Position Embeddings) encodes positional information by
          rotating query and key vectors in the embedding space, allowing the
          model to capture relative positions naturally. RoPE can be combined
          with GQA if needed, but they are independent features.
    """
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    max_seq_len = 2048
    return [dropout_p, is_causal, enable_gqa, max_seq_len]