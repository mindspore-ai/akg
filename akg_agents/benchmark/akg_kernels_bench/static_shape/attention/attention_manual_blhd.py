import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    一个简化的 FlashAttention 算子实现。
    注意：真实的 FlashAttention 需要复杂的 CUDA 内核进行 tiling 和在线 softmax 重计算以优化显存。
    此实现使用 PyTorch 标准函数模拟其前向计算逻辑，作为演示之用。
    """
    def __init__(self, dropout_p=0.0, causal=False, softmax_scale=None):
        super().__init__()
        self.dropout_p = dropout_p
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(self, q, k, v, attn_mask=None):
        """
        Forward pass simulating FlashAttention computation steps.
        
        IMPORTANT: This implementation uses (B, L, H, D) layout, which is DIFFERENT from
        torch.nn.functional.scaled_dot_product_attention's required (B, H, L, D) layout.
        
        This layout (B, L, H, D) is more intuitive as it follows "batch -> sequence -> heads"
        order, but requires transpose operations for PyTorch's optimized SDPA function.
        
        Args:
            q: Query tensor of shape (B, L, H, D) where:
               - B: Batch size
               - L: Sequence length for query
               - H: Number of attention heads
               - D: Head dimension
            k: Key tensor of shape (B, S, H, D) where S is key/value sequence length
            v: Value tensor of shape (B, S, H, D)
            attn_mask: Optional attention mask, shape (B, L, S) or (L, S)
            
        Returns:
            out: Output tensor of shape (B, L, H, D)
        """
        # 保存原始形状用于恢复
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        seq_len_kv = k.shape[1]
        
        # 重塑 Q, K, V 以进行批量矩阵乘法 (BMM)
        # 合并 batch 和 heads 维度: (b, s, h, d) -> (b*h, s, d)
        q_reshaped = q.transpose(1, 2).reshape(batch_size * num_heads, seq_len_q, head_dim)
        k_reshaped = k.transpose(1, 2).reshape(batch_size * num_heads, seq_len_kv, head_dim)
        v_reshaped = v.transpose(1, 2).reshape(batch_size * num_heads, seq_len_kv, head_dim)
        
        # 计算缩放点积注意力分数: (b*h, s_q, d) @ (b*h, d, s_kv) -> (b*h, s_q, s_kv)
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2))
        
        # 应用缩放因子 (默认是 1 / sqrt(head_dim))
        scale_factor = self.softmax_scale if self.softmax_scale is not None else (head_dim ** -0.5)
        attn_scores = attn_scores * scale_factor
        
        # 应用因果掩码 (用于自回归解码)
        if self.causal:
            # 创建一个上三角掩码 (主对角线及以下为 True，以上为 False)
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=q.device), diagonal=1)
            attn_scores = attn_scores.view(batch_size, num_heads, seq_len_q, seq_len_kv)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            attn_scores = attn_scores.view(batch_size * num_heads, seq_len_q, seq_len_kv)
        
        # 应用外部提供的注意力掩码
        if attn_mask is not None:
            # 处理掩码形状: 可能是 (b, s_q, s_kv) 或 (s_q, s_kv)
            # 需要广播到 (b*h, s_q, s_kv)
            attn_mask = attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.expand(batch_size, num_heads, seq_len_q, seq_len_kv)
            attn_mask = attn_mask.reshape(batch_size * num_heads, seq_len_q, seq_len_kv)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        # 计算 Softmax 得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用 Dropout
        if self.dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        
        # 加权求和: (b*h, s_q, s_kv) @ (b*h, s_kv, d) -> (b*h, s_q, d)
        out = torch.bmm(attn_weights, v_reshaped)
        
        # 恢复原始形状: (b*h, s_q, d) -> (b, s_q, h, d)
        out = out.view(batch_size, num_heads, seq_len_q, head_dim).transpose(1, 2)
        
        return out

def get_inputs():
    """
    Generate input tensors for FlashAttention with (B, L, H, D) layout.
    
    IMPORTANT: This uses (B, L, H, D) layout, NOT the standard (B, H, L, D) layout
    used by torch.nn.functional.scaled_dot_product_attention.
    
    Tensor shape: (B, L, H, D)
        B = 1    : Batch size
        L = 256  : Sequence length for query
        S = 256  : Sequence length for key/value (same as L in this case)
        H = 8    : Number of attention heads
        D = 64   : Head dimension (embedding size per head)
    
    Total model dimension = H * D = 8 * 64 = 512
    
    Note: For inference scenarios, consider using torch.no_grad() context
          to disable gradient computation and save memory.
    """
    B, L, H, D = 1, 256, 8, 64
    S = 256  # Key/value sequence length (can differ from L)
    
    # Use smaller std to avoid float16 overflow in attention computation
    q = torch.empty(B, L, H, D, dtype=torch.float16).normal_(mean=0.0, std=0.1)
    k = torch.empty(B, S, H, D, dtype=torch.float16).normal_(mean=0.0, std=0.1)
    v = torch.empty(B, S, H, D, dtype=torch.float16).normal_(mean=0.0, std=0.1)
    
    # Optional: Create a causal mask (upper triangle is True, masking future tokens)
    # causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device='cuda'), diagonal=1)
    # return [q, k, v, causal_mask]
    
    # Return without mask
    return [q, k, v, None]

def get_init_inputs():
    """
    Return initialization parameters for the Model class.
    
    Returns empty list to use default values:
        - dropout_p=0.0 (no dropout for inference)
        - causal=False (not using causal masking)
        - softmax_scale=None (will use 1/sqrt(head_dim) by default)
    """
    return []