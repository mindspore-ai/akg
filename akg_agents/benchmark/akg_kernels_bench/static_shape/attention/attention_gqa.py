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
        Grouped-Query Attention (GQA) using PyTorch's scaled dot-product attention.
        
        GQA is a compromise between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
        It groups multiple query heads to share key-value heads, balancing quality and efficiency.
        
        Tensor layout: (B, H, L, D) where:
            - B (Batch): Number of sequences processed in parallel
            - H (Heads): Number of attention heads
            - L (Length): Sequence length (number of tokens)
            - D (Dimension): Embedding dimension per head
        
        For GQA specifically:
            - Query shape: (B, H_q, L, D) where H_q = number of query heads
            - Key shape:   (B, H_kv, S, D) where H_kv = number of key/value heads (H_kv < H_q)
            - Value shape: (B, H_kv, S, D)
        
        Each key/value head is shared across H_q/H_kv query heads.
        Example: 8 query heads with 4 KV heads means each KV head serves 2 query heads.
        
        Args:
            query: Query tensor of shape (B, H_q, L, D)
            key: Key tensor of shape (B, H_kv, S, D) where H_kv < H_q
            value: Value tensor of shape (B, H_kv, S, D)
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
    Generate input tensors for Grouped-Query Attention (GQA).
    
    GQA uses different head counts for queries vs keys/values:
    
    Query shape: (B, H_q, L, D)
        B = 32   : Batch size
        H_q = 8  : Number of query heads
        L = 1024 : Sequence length
        D = 64   : Head dimension
    
    Key/Value shape: (B, H_kv, L, D)
        B = 32    : Batch size (same as query)
        H_kv = 4  : Number of key/value heads (4 groups, each serving 2 query heads)
        L = 1024  : Sequence length (same as query)
        D = 64    : Head dimension (same as query)
    
    Memory savings: GQA reduces KV cache from (B, 8, L, D) to (B, 4, L, D),
    which is 2x smaller. This balances quality (better than MQA) and efficiency
    (better than MHA).
    
    Comparison:
        - MHA (Multi-Head): H_q = H_kv = 8 (no sharing)
        - GQA (Grouped):    H_q = 8, H_kv = 4 (2:1 sharing ratio)
        - MQA (Multi-Query): H_q = 8, H_kv = 1 (8:1 sharing ratio)
    """
    B, H_q, L, D = 32, 8, 1024, 64
    H_kv = 4  # 4 key/value heads for 8 query heads (2:1 ratio)
    
    query_shape = (B, H_q, L, D)
    key_value_shape = (B, H_kv, L, D)
    
    # Use smaller std to avoid float16 overflow in attention computation
    query = torch.empty(query_shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    key = torch.empty(key_value_shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    value = torch.empty(key_value_shape, dtype=torch.float16).normal_(mean=0.5, std=0.1)
    return [query, key, value]


def get_init_inputs():
    """
    Initialize parameters for Grouped-Query Attention.
    
    Returns:
        dropout_p: 0.0 (no dropout for inference)
        is_causal: False (not using causal masking)
        enable_gqa: True (enable GQA optimizations in PyTorch)
    """
    dropout_p = 0.0
    is_causal = False
    enable_gqa = True
    return [dropout_p, is_causal, enable_gqa]