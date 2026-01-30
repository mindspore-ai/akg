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
        前向传播，模拟 FlashAttention 的计算步骤。
        Args:
            q: [batch_size, seq_len_q, num_heads, head_dim]
            k: [batch_size, seq_len_kv, num_heads, head_dim]
            v: [batch_size, seq_len_kv, num_heads, head_dim]
            attn_mask: 可选，[batch_size, seq_len_q, seq_len_kv] 或 [seq_len_q, seq_len_kv]
        Returns:
            out: [batch_size, seq_len_q, num_heads, head_dim]
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
    返回模型 forward 方法所需的输入 tensor 列表。
    使用合理的默认形状。
    """
    batch_size = 1
    seq_len_q = 256
    seq_len_kv = 256
    num_heads = 8
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device='cuda')
    v = torch.randn(batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device='cuda')
    
    # 可选：创建一个因果掩码 (上三角为 True，表示需要屏蔽未来信息)
    # causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device='cuda'), diagonal=1)
    # return [q, k, v, causal_mask]
    
    # 本次返回不带掩码
    return [q, k, v, None]

def get_init_inputs():
    """
    返回 Model 类初始化所需的参数列表。
    这里返回空列表，使用默认的 dropout_p=0.0 和 causal=False。
    """
    return []