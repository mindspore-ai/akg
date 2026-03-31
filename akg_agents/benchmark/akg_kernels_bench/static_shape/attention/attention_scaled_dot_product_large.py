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
        Scaled Dot-Product Attention with large tensor sizes for production workloads.
        
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
    Generate large input tensors for production-scale attention benchmarking.
    
    Tensor shape: (B, H, L, D)
        B = 64   : Batch size (large batch for throughput)
        H = 32   : Number of attention heads (typical for large models)
        L = 2048 : Sequence length (long context window)
        D = 128  : Head dimension (larger than typical 64)
    
    Total model dimension = H * D = 32 * 128 = 4096
    
    Note: These large sizes are representative of:
          - Large language models (LLMs) like GPT-3, LLaMA
          - Long-context applications (2K+ tokens)
          - High-throughput inference scenarios
          - Memory and performance stress testing
    
    Warning: This configuration requires significant GPU memory (multiple GBs).
    """
    B, H, L, D = 64, 32, 2048, 128
    shape = (B, H, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    query = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    key = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    value = torch.empty(shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for large-scale attention benchmarking.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: False (not using causal masking)
        enable_gqa: False (standard multi-head attention)
    """
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]