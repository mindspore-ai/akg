import torch
import torch.nn as nn
import math


def _bhld(B, H, L, D, dtype=torch.float16):
    """Match static_shape/attention/attention_rope.py distribution."""
    q = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    k = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    v = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    return q, k, v


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False, max_seq_len=2048):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.max_seq_len = max_seq_len
        self.enable_gqa = enable_gqa

    def forward(self, query, key, value, attn_mask=None):
        query_rot = self._apply_rotary_embeddings(query)
        key_rot = self._apply_rotary_embeddings(key)
        return torch.nn.functional.scaled_dot_product_attention(
            query_rot, key_rot, value, attn_mask=attn_mask,
            dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )

    def _apply_rotary_embeddings(self, x):
        batch, num_heads, seq_len, head_dim = x.shape
        pos = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        dim = head_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(dim, dtype=x.dtype, device=x.device) / dim)
        freqs = pos * freqs.unsqueeze(0)
        cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.cat([
            x_even * cos_freqs - x_odd * sin_freqs,
            x_odd * cos_freqs + x_even * sin_freqs
        ], dim=-1)
        return x_rot


def get_inputs_dyn_list():
    """Same as static rope but multiple (B,H,L,D); head_dim even for RoPE split."""
    dtype = torch.float16
    q1, k1, v1 = _bhld(4, 2, 64, 16, dtype=dtype)
    q2, k2, v2 = _bhld(4, 2, 64, 16, dtype=dtype)
    q3, k3, v3 = _bhld(8, 4, 128, 32, dtype=dtype)
    q4, k4, v4 = _bhld(8, 4, 128, 32, dtype=dtype)
    q5, k5, v5 = _bhld(16, 8, 256, 64, dtype=dtype)
    return [[q1, k1, v1], [q2, k2, v2], [q3, k3, v3], [q4, k4, v4], [q5, k5, v5]]


def get_init_inputs():
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    max_seq_len = 2048
    return [dropout_p, is_causal, enable_gqa, max_seq_len]
