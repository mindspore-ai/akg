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
        Scaled Dot-Product Attention with small tensor sizes for testing/debugging.
        
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
    Generate small input tensors for testing/debugging attention.
    
    Tensor shape: (B, H, L, D)
        B = 8  : Batch size (small for quick testing)
        H = 4  : Number of attention heads
        L = 32 : Sequence length (short sequences)
        D = 32 : Head dimension (smaller than typical 64)
    
    Total model dimension = H * D = 4 * 32 = 128
    
    Note: These small sizes are useful for:
          - Quick unit testing and debugging
          - Verifying correctness before scaling up
          - Running on devices with limited memory
    """
    B, H, L, D = 8, 4, 32, 32
    shape = (B, H, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    query = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    key = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    value = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for small-scale attention testing.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: False (not using causal masking)
        enable_gqa: False (standard multi-head attention)
    """
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]