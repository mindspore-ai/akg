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
        Multi-Query Attention (MQA) using PyTorch's scaled dot-product attention.
        
        MQA is a memory-efficient variant where multiple query heads share a single
        key-value head. This reduces the KV cache size significantly, which is crucial
        for inference with long sequences.
        
        Tensor layout: (B, H, L, D) where:
            - B (Batch): Number of sequences processed in parallel
            - H (Heads): Number of attention heads
            - L (Length): Sequence length (number of tokens)
            - D (Dimension): Embedding dimension per head
        
        For MQA specifically:
            - Query shape: (B, H_q, L, D) where H_q = number of query heads
            - Key shape:   (B, 1, S, D) where 1 = single shared key head
            - Value shape: (B, 1, S, D) where 1 = single shared value head
        
        The single key/value head is automatically broadcasted across all query heads.
        
        Args:
            query: Query tensor of shape (B, H_q, L, D)
            key: Key tensor of shape (B, 1, S, D) - single shared head
            value: Value tensor of shape (B, 1, S, D) - single shared head
            attn_mask: Optional attention mask
            
        Returns:
            Attention output of shape (B, H_q, L, D)
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs():
    """
    Generate input tensors for Multi-Query Attention (MQA).
    
    MQA uses different head counts for queries vs keys/values:
    
    Query shape: (B, H_q, L, D)
        B = 32   : Batch size
        H_q = 8  : Number of query heads
        L = 1024 : Sequence length
        D = 64   : Head dimension
    
    Key/Value shape: (B, H_kv, L, D)
        B = 32    : Batch size (same as query)
        H_kv = 1  : Number of key/value heads (1 for MQA, shared across all query heads)
        L = 1024  : Sequence length (same as query)
        D = 64    : Head dimension (same as query)
    
    Memory savings: MQA reduces KV cache from (B, 8, L, D) to (B, 1, L, D),
    which is 8x smaller - crucial for long-context inference.
    
    Note: For Grouped-Query Attention (GQA), H_kv would be > 1 but < H_q.
    """
    B, H_q, L, D = 32, 8, 1024, 64
    H_kv = 1  # Single shared key/value head for MQA
    
    query_shape = (B, H_q, L, D)
    key_value_shape = (B, H_kv, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    query = torch.empty(query_shape, dtype=torch.float16).normal_(mean=0.0, std=0.1)
    key = torch.empty(key_value_shape, dtype=torch.float16).normal_(mean=0.0, std=0.1)
    value = torch.empty(key_value_shape, dtype=torch.float16).normal_(mean=0.0, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for Multi-Query Attention.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: False (not using causal masking)
        enable_gqa: False (MQA is handled via tensor shapes, not this flag)
    
    Note: MQA behavior is determined by the key/value tensor shapes (H_kv=1),
          not by the enable_gqa flag. The enable_gqa flag is for PyTorch's
          internal optimizations for grouped-query attention patterns.
    """
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]