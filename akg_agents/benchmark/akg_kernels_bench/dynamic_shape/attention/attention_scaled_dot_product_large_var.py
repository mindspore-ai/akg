import torch
import torch.nn as nn


def _bhld(B, H, L, D, dtype=torch.float16):
    """Match static_shape input distribution."""
    q = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    k = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    v = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    return q, k, v


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

    def forward(self, query, key, value, attn_mask=None):
        """Same as static_shape/attention/attention_scaled_dot_product_large.py."""
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs_dyn_list():
    """Large (B,H,L,D); same spirit as static large (64,32,2048,128)."""
    dtype = torch.float16
    cases = [
        (8, 16, 512, 128),
        (16, 32, 1024, 128),
        (32, 32, 1024, 128),
        (64, 32, 1024, 128),
        (64, 32, 2048, 128),
    ]
    out = []
    for B, H, L, D in cases:
        q, k, v = _bhld(B, H, L, D, dtype=dtype)
        out.append([q, k, v])
    return out


def get_init_inputs():
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]
