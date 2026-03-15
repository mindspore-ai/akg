import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Multi-Latent Attention (MLA) Decode with Paged KV Cache (PyTorch Reference).
    
    Used in DeepSeek-V3 for efficient attention with compressed KV cache.
    Separates K/Q into two components:
        - K/Q_nope: Content-aware features without rotation
        - K/Q_rope: Positional features with RoPE applied
    
    Attention score: (Q_nope @ K_nope.T) + (Q_rope @ K_rope.T)
    """

    def __init__(self, sm_scale, page_size):
        super(Model, self).__init__()
        self.sm_scale = sm_scale
        self.page_size = page_size

    def forward(self, q, k_nope_buffer, k_rope_buffer, v_buffer, seq_lens, block_table):
        """
        Args:
            q: (batch, num_heads, qk_nope_dim + qk_rope_dim) - query tensor
            k_nope_buffer: (num_pages, page_size, num_kv_heads, qk_nope_dim) - K_nope cache
            k_rope_buffer: (num_pages, page_size, num_kv_heads, qk_rope_dim) - K_rope cache
            v_buffer: (num_pages, page_size, num_kv_heads, v_dim) - V cache
            seq_lens: (batch,) - sequence lengths
            block_table: (batch, max_pages) - page table for KV cache
        
        Returns:
            attn_out: (batch, num_heads, v_dim) - attention output
        """
        batch = q.shape[0]
        num_q_heads = q.shape[1]
        qk_nope_dim = k_nope_buffer.shape[-1]
        qk_rope_dim = k_rope_buffer.shape[-1]
        num_kv_heads = k_nope_buffer.shape[2]
        v_dim = v_buffer.shape[-1]
        
        outputs = []

        for i in range(batch):
            kv_len = seq_lens[i].item()

            # Split Q into Q_nope and Q_rope
            q_i = q[i:i+1]  # (1, num_heads, qk_dim)
            q_nope = q_i[:, :, :qk_nope_dim]
            q_rope = q_i[:, :, qk_nope_dim:]

            # Gather K and V from paged cache
            num_blocks = (kv_len + self.page_size - 1) // self.page_size
            block_ids = block_table[i, :num_blocks]
            
            k_nope = k_nope_buffer[block_ids].view(-1, num_kv_heads, qk_nope_dim)[:kv_len]
            k_rope = k_rope_buffer[block_ids].view(-1, num_kv_heads, qk_rope_dim)[:kv_len]
            v = v_buffer[block_ids].view(-1, num_kv_heads, v_dim)[:kv_len]
            
            # GQA: repeat KV heads if needed
            if num_q_heads != num_kv_heads:
                rep_factor = num_q_heads // num_kv_heads
                k_nope = torch.repeat_interleave(k_nope, rep_factor, dim=1)
                k_rope = torch.repeat_interleave(k_rope, rep_factor, dim=1)
                v = torch.repeat_interleave(v, rep_factor, dim=1)
            
            # Compute attention scores: QK_nope + QK_rope
            qk_nope = torch.einsum("qhd,khd->hqk", q_nope, k_nope).float()
            qk_rope = torch.einsum("qhd,khd->hqk", q_rope, k_rope).float()
            qk = (qk_nope + qk_rope) * self.sm_scale
            
            # Softmax and attention output
            score = torch.softmax(qk, dim=-1).to(v.dtype)
            out_i = torch.einsum("hqk,khd->qhd", score, v)
            
            outputs.append(out_i)
        
        return torch.cat(outputs, dim=0)


def get_inputs():
    batch = 16
    seq_len = 1024
    num_q_heads = 128
    num_kv_heads = 1
    qk_nope_dim = 512
    qk_rope_dim = 64
    v_dim = 512
    page_size = 128
    dtype = torch.float32
    
    max_pages = (seq_len + page_size - 1) // page_size
    
    q = torch.randn(
        batch, num_q_heads, qk_nope_dim + qk_rope_dim,
        dtype=dtype
    )
    k_nope_buffer = torch.randn(
        max_pages * batch, page_size, num_kv_heads, qk_nope_dim,
        dtype=dtype
    )
    k_rope_buffer = torch.randn(
        max_pages * batch, page_size, num_kv_heads, qk_rope_dim,
        dtype=dtype
    )
    v_buffer = torch.randn(
        max_pages * batch, page_size, num_kv_heads, v_dim,
        dtype=dtype
    )
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    block_table = torch.arange(
        0, batch * max_pages, dtype=torch.int32
    ).reshape(batch, max_pages)
    
    return [q, k_nope_buffer, k_rope_buffer, v_buffer, seq_lens, block_table]


def get_init_inputs():
    qk_dim = 576  # qk_nope_dim + qk_rope_dim
    sm_scale = 1.0 / (qk_dim ** 0.5)
    page_size = 128
    return [sm_scale, page_size]
