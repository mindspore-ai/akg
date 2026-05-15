import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Same implementation as static_shape/attention/attention_manual_blhd.py (BLHD layout)."""

    def __init__(self, dropout_p=0.0, causal=False, softmax_scale=None):
        super().__init__()
        self.dropout_p = dropout_p
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(self, q, k, v, attn_mask=None):
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        seq_len_kv = k.shape[1]
        q_reshaped = q.transpose(1, 2).reshape(batch_size * num_heads, seq_len_q, head_dim)
        k_reshaped = k.transpose(1, 2).reshape(batch_size * num_heads, seq_len_kv, head_dim)
        v_reshaped = v.transpose(1, 2).reshape(batch_size * num_heads, seq_len_kv, head_dim)
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2))
        scale_factor = self.softmax_scale if self.softmax_scale is not None else (head_dim ** -0.5)
        attn_scores = attn_scores * scale_factor
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=q.device), diagonal=1)
            attn_scores = attn_scores.view(batch_size, num_heads, seq_len_q, seq_len_kv)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            attn_scores = attn_scores.view(batch_size * num_heads, seq_len_q, seq_len_kv)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.expand(batch_size, num_heads, seq_len_q, seq_len_kv)
            attn_mask = attn_mask.reshape(batch_size * num_heads, seq_len_q, seq_len_kv)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        out = torch.bmm(attn_weights, v_reshaped)
        out = out.view(batch_size, num_heads, seq_len_q, head_dim).transpose(1, 2)
        return out


def _blhd(B, L, H, D, S=None, dtype=torch.float16):
    """Same distribution as static manual_blhd get_inputs."""
    if S is None:
        S = L
    q = torch.empty(B, L, H, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    k = torch.empty(B, S, H, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    v = torch.empty(B, S, H, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    return q, k, v


def get_inputs_dyn_list():
    """(B, L, H, D) / (B, S, H, D); same case idea as BHLD sweeps, BLHD layout."""
    dtype = torch.float16
    specs = [
        (1, 32, 8, 64, 32),
        (1, 64, 8, 64, 64),
        (2, 128, 8, 64, 128),
        (4, 256, 8, 64, 256),
    ]
    out = []
    for B, L, H, D, S in specs:
        q, k, v = _blhd(B, L, H, D, S=S, dtype=dtype)
        out.append([q, k, v, None])
    return out


def get_init_inputs():
    return []
