import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

    def forward(self, query, key, value, attn_mask=None):
        """
        Flash Attention with Causal Masking for autoregressive generation.
        
        Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + causal_mask) @ V
        
        Causal masking ensures that each position can only attend to previous positions
        and itself, preventing information leakage from future tokens. This is essential
        for autoregressive models like GPT.
        
        The causal mask is a lower-triangular matrix:
            [[0, -inf, -inf, -inf],
             [0,    0, -inf, -inf],
             [0,    0,    0, -inf],
             [0,    0,    0,    0]]
        
        Input tensor layout: (B, H, L, D) where:
            - B (Batch): Number of sequences processed in parallel
            - H (Heads): Number of attention heads
            - L (Length): Sequence length (number of tokens)
            - D (Dimension): Embedding dimension per head
        
        Args:
            query: Query tensor of shape (B, H, L, D)
            key: Key tensor of shape (B, H, S, D), where S can differ from L
            value: Value tensor of shape (B, H, S, D)
            attn_mask: Optional additional attention mask (combined with causal mask)
            
        Returns:
            Attention output of shape (B, H, L, D)
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs():
    """
    Generate input tensors for causal flash attention.
    
    Tensor shape: (B, H, L, D)
        B = 16  : Batch size (number of independent sequences)
        H = 12  : Number of attention heads (multi-head attention)
        L = 512 : Sequence length (number of tokens in each sequence)
        D = 64  : Head dimension (embedding size per attention head)
    
    Total model dimension = H * D = 12 * 64 = 768
    
    Note: Causal masking is commonly used in decoder-only models (GPT-style)
          for autoregressive text generation.
    """
    B, H, L, D = 16, 12, 512, 64
    shape = (B, H, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    query = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    key = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    value = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for causal flash attention.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: True (enable causal masking for autoregressive generation)
        enable_gqa: False (standard multi-head attention)
    """
    dropout_p = 0.0
    is_causal = True
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]