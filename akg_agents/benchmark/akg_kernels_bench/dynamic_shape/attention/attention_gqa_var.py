import torch
import torch.nn as nn


def _gqa(B, H_q, H_kv, L, D, dtype=torch.float16):
    """Match static_shape/attention/attention_gqa.py distribution."""
    q = torch.empty(B, H_q, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    k = torch.empty(B, H_kv, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    v = torch.empty(B, H_kv, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    return q, k, v


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

    def forward(self, query, key, value, attn_mask=None):
        """Same as static_shape/attention/attention_gqa.py (GQA)."""
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs_dyn_list():
    """H_q > H_kv; same head ratios as previous dynamic gqa cases."""
    dtype = torch.float16
    q1, k1, v1 = _gqa(4, 4, 2, 64, 16, dtype=dtype)
    q2, k2, v2 = _gqa(4, 4, 2, 64, 16, dtype=dtype)
    q3, k3, v3 = _gqa(8, 8, 4, 128, 32, dtype=dtype)
    q4, k4, v4 = _gqa(8, 8, 4, 128, 32, dtype=dtype)
    q5, k5, v5 = _gqa(16, 16, 8, 256, 64, dtype=dtype)
    return [[q1, k1, v1], [q2, k2, v2], [q3, k3, v3], [q4, k4, v4], [q5, k5, v5]]


def get_init_inputs():
    dropout_p = 0.0
    is_causal = False
    enable_gqa = True
    return [dropout_p, is_causal, enable_gqa]
