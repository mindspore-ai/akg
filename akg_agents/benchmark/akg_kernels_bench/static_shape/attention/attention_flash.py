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
        Flash Attention using PyTorch's optimized scaled dot-product attention.
        
        Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        PyTorch automatically dispatches to FlashAttention implementation when available,
        which uses tiling and online softmax to reduce memory usage from O(N²) to O(N).
        
        Input tensor layout: (B, H, L, D) where:
            - B (Batch): Number of sequences processed in parallel
            - H (Heads): Number of attention heads
            - L (Length): Sequence length (number of tokens)
            - D (Dimension): Embedding dimension per head
        
        Args:
            query: Query tensor of shape (B, H, L, D)
            key: Key tensor of shape (B, H, S, D), where S can differ from L
            value: Value tensor of shape (B, H, S, D)
            attn_mask: Optional attention mask
            
        Returns:
            Attention output of shape (B, H, L, D)
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs():
    """
    Generate input tensors for flash attention.
    
    Tensor shape: (B, H, L, D)
        B = 32   : Batch size (number of independent sequences)
        H = 8    : Number of attention heads (multi-head attention)
        L = 1024 : Sequence length (number of tokens in each sequence)
        D = 64   : Head dimension (embedding size per attention head)
    
    Total model dimension = H * D = 8 * 64 = 512
    
    Note: Using torch.randn() which doesn't require gradients by default.
          For training, you would need to set requires_grad=True.
    """
    B, H, L, D = 32, 8, 1024, 64
    shape = (B, H, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    # Shift mean to 0.5 to avoid output being too close to 0, which causes relative error explosion
    query = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    key = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    value = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for flash attention.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: False (not using causal masking)
        enable_gqa: False (standard multi-head attention)
    """
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]