import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, g, beta, initial_state, orig_mask):
        initial_dtype = query.dtype
        chunk_size=64
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]

        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        # reshape to chunks
        # (batch, heads, seq_len, dim) → (batch, heads, num_chunks, chunk_size, dim)
        # 门控制向量：(batch, heads, seq_len) → (batch, heads, num_chunks, chunk_size)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

        g = g.cumsum(dim=-1)
        # g.unsqueeze(-1)：(batch, heads, num_chunks, chunk_size) → (batch, heads, num_chunks, chunk_size, 1)
        # g.unsqueeze(-2)：(batch, heads, num_chunks, chunk_size) → (batch, heads, num_chunks, 1, chunk_size)
        # 减法：(batch, heads, num_chunks, chunk_size, chunk_size)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

        mask = torch.triu(orig_mask, diagonal=0)
        # [batch_size, num_heads, num_chunks, chunk_size, k_head_dim] @ [batch_size, num_heads, num_chunks, k_head_dim, chunk_size]
        # = [batch_size, num_heads, num_chunks, chunk_size, chunk_size]
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        # attn：[batch_size, num_heads, num_chunks, chunk_size, chunk_size]
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        # [batch_size, num_heads, num_chunks, chunk_size, chunk_size] @ [batch_size, num_heads, num_chunks, chunk_size, v_head_dim]
        # = [batch_size, num_heads, num_chunks, chunk_size, v_head_dim]
        value = attn @ v_beta
        # [batch_size, num_heads, num_chunks, chunk_size, chunk_size] @ [batch_size, num_heads, num_chunks, chunk_size, k_head_dim]
        # = [batch_size, num_heads, num_chunks, chunk_size, k_head_dim]
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        # ​​循环状态初始化​，(batch, heads, k_head_dim, v_head_dim)
        last_recurrent_state = initial_state
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(orig_mask, diagonal=1)

        # for each chunk
        # 分块循环处理​
        for i in range(0, sequence_length  // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            # [batch_size, num_heads, 1, chunk_size, k_head_dim] @ [batch_size, num_heads, 1, k_head_dim, chunk_size]
            # = [batch_size, num_heads, 1, chunk_size, chunk_size]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)

            # [batch_size, num_heads, 1, chunk_size, k_head_dim] @ [batch_size, num_heads, k_head_dim, v_head_dim]
            # = [batch_size, num_heads, 1, chunk_size, v_head_dim]
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime

            # [batch_size, num_heads, 1, chunk_size, k_head_dim] @ [batch_size, num_heads, k_head_dim, v_head_dim]
            # = [batch_size, num_heads, 1, chunk_size, v_head_dim]
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            # [batch_size, num_heads, 1, chunk_size, chunk_size] @ [batch_size, num_heads, 1, chunk_size, v_head_dim]
            # = [batch_size, num_heads, 1, chunk_size, v_head_dim]
            core_attn_out[:, :, i] = attn_inter + attn @ v_new

            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

def get_inputs():
    batch, num_heads, seq_len, head_dim = 32, 8, 1024, 64
    chunk_size = 64
    shape = (batch, num_heads, seq_len, head_dim)
    
    query = torch.randn(shape, dtype=torch.float32)
    key = torch.randn(shape, dtype=torch.float32)
    value = torch.randn(shape, dtype=torch.float32)
    g = torch.randn((batch, num_heads, seq_len), dtype=torch.float32)
    beta = torch.randn((batch, num_heads, seq_len), dtype=torch.float32)

    initial_state = torch.zeros(batch, num_heads, head_dim, head_dim, dtype=torch.float32)
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool)
    return [query, key, value, g, beta, initial_state, mask]

def get_init_inputs():
    return []