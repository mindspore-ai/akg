import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, g, beta, initial_state):
        initial_dtype = query.dtype
        # 转置时确保维度正确
        # query = query.transpose(1, 2).contiguous().to(torch.float32)
        # key = key.transpose(1, 2).contiguous().to(torch.float32)
        # value = value.transpose(1, 2).contiguous().to(torch.float32)
        # beta = beta.transpose(1, 2).contiguous().to(torch.float32)
        # g = g.transpose(1, 2).contiguous().to(torch.float32)
        
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=torch.float32)
        
        # 正确初始化状态张量
        if initial_state is None:
            last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        else:
            last_recurrent_state = initial_state.to(value.device).to(torch.float32)

        for i in range(sequence_length):
            q_t = query[:, :, i]  # [batch, heads, 1, dim]
            k_t = key[:, :, i]    # [batch, heads, 1, dim]
            v_t = value[:, :, i]  # [batch, heads, 1, dim]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [batch, heads, 1, 1]
            beta_t = beta[:, :, i].unsqueeze(-1)  # [batch, heads, 1]

            # 更新循环状态
            # batch_size, num_heads, k_head_dim, v_head_dim
            last_recurrent_state = last_recurrent_state * g_t
            
            # 修复维度不匹配问题：确保k_t有正确的维度
            k_t_expanded = k_t.unsqueeze(-1)  # [batch, heads, dim, 1]
            
            # 计算kv_mem
            # batch_size, num_heads, 1, v_head_dim
            kv_mem = (last_recurrent_state * k_t_expanded).sum(dim=-2)  # [batch, heads, v_dim]
            
            # 计算delta并更新状态
            delta = (v_t - kv_mem) * beta_t
            delta_expanded = delta.unsqueeze(-2)  # [batch, heads, 1, v_dim]
            # batch_size, num_heads, k_head_dim, v_head_dim
            last_recurrent_state = last_recurrent_state + k_t_expanded * delta_expanded
            
            # 计算输出
            q_t_expanded = q_t.unsqueeze(-1)  # [batch, heads, dim, 1]
            # batch_size, num_heads, 1, v_head_dim
            core_attn_out[:, :, i] = (last_recurrent_state * q_t_expanded).sum(dim=-2)

        # batch_size, sequence_length, num_heads, v_head_dim
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

def get_inputs():
    batch, num_heads, seq_len, head_dim = 32, 8, 1024, 64
    shape = (batch, num_heads, seq_len, head_dim)
    
    query = torch.randn(shape, dtype=torch.float32)
    key = torch.randn(shape, dtype=torch.float32)
    value = torch.randn(shape, dtype=torch.float32)
    g = torch.randn((batch, num_heads, seq_len), dtype=torch.float32)
    beta = torch.randn((batch, num_heads, seq_len), dtype=torch.float32)
    return [query, key, value, g, beta, None]

def get_init_inputs():
    return []