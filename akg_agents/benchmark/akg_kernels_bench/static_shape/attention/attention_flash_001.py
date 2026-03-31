import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Scaled Dot-Product Attention implementation using PyTorch's optimized function.
        
        This function computes: Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V
        
        Input tensor layout (required by torch.nn.functional.scaled_dot_product_attention):
            Shape: (B, H, L, D) where:
            - B (Batch): Number of sequences processed in parallel
            - H (Heads): Number of attention heads
            - L (Length): Sequence length (number of tokens)
            - D (Dimension): Embedding dimension per head
            
        Note: The sequence length MUST be at dimension -2 (second to last),
              and embedding dimension MUST be at dimension -1 (last).
        
        Args:
            query: Query tensor of shape (B, H, L, D)
            key: Key tensor of shape (B, H, S, D), where S can differ from L
            value: Value tensor of shape (B, H, S, D)
            
        Returns:
            Attention output of shape (B, H, L, D)
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            scale=self.sm_scale
        )


def get_inputs():
    """
    Generate input tensors for scaled dot-product attention.
    
    Tensor shape: (B, H, L, D)
        B = 4     : Batch size (number of independent sequences)
        H = 32    : Number of attention heads (multi-head attention)
        L = 1024  : Sequence length (number of tokens in each sequence)
        D = 64    : Head dimension (embedding size per attention head)
    
    Total model dimension = H * D = 32 * 64 = 2048
    
    Note: For inference, we don't need gradient computation, so .requires_grad_()
          is removed. Use torch.no_grad() context or set requires_grad=False.
    """
    B, H, L, D = 4, 32, 1024, 64
    dtype = torch.float16
    
    # Use smaller std to avoid float16 overflow in attention computation
    q = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.5, std=0.1)
    k = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.5, std=0.1)
    v = torch.empty((B, H, L, D), dtype=dtype).normal_(mean=0.5, std=0.1)
    
    return [q, k, v]


def get_init_inputs():
    """
    Initialize the scaling factor for attention.
    
    Standard scaling factor is 1/sqrt(D) = 1/sqrt(64) ≈ 0.125
    Here we use 0.5 for demonstration purposes.
    """
    sm_scale = 0.5
    return [sm_scale]