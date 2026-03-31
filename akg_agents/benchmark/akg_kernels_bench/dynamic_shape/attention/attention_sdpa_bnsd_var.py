import torch
import torch.nn as nn


def _bhld(B, H, L, D, dtype=torch.float16):
    """Match static_shape/attention input distribution (avoid near-zero outputs)."""
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
        """
        Scaled Dot-Product Attention with (B, H, L, D) layout (BNSD naming).

        Same as static_shape/attention/attention_sdpa_bnsd.py.
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs_dyn_list():
    """Same 10 cases as attention_flash_var (BHLD layout); only shapes vary."""
    q1, k1, v1 = _bhld(1, 1, 15, 15)
    q2, k2, v2 = _bhld(1, 1, 31, 31)
    q3, k3, v3 = _bhld(1, 1, 32, 32)
    q4, k4, v4 = _bhld(4, 4, 63, 63)
    q5, k5, v5 = _bhld(4, 4, 64, 64)
    q6, k6, v6 = _bhld(8, 8, 127, 127)
    q7, k7, v7 = _bhld(8, 8, 128, 128)
    q8, k8, v8 = _bhld(16, 16, 255, 255)
    q9, k9, v9 = _bhld(16, 16, 256, 256)
    q10, k10, v10 = _bhld(32, 8, 511, 64)
    return [
        [q1, k1, v1], [q2, k2, v2], [q3, k3, v3], [q4, k4, v4], [q5, k5, v5],
        [q6, k6, v6], [q7, k7, v7], [q8, k8, v8], [q9, k9, v9], [q10, k10, v10],
    ]


def get_init_inputs():
    dropout_p = 0.0
    is_causal = False
    enable_gqa = False
    return [dropout_p, is_causal, enable_gqa]
