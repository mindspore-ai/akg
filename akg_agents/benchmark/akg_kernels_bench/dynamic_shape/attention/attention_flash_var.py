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
        Flash Attention using PyTorch's optimized scaled dot-product attention.

        Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Input tensor layout: (B, H, L, D)
        """
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=self.is_causal,
            enable_gqa=self.enable_gqa
        )


def get_inputs_dyn_list():
    """Same case IDs as other BHLD SDPA *_var modules; only (B,H,L,D) vary."""
    # Case 1: (1, 1, 15, 15) non-aligned
    q1, k1, v1 = _bhld(1, 1, 15, 15)
    # Case 2: (1, 1, 31, 31) non-aligned
    q2, k2, v2 = _bhld(1, 1, 31, 31)
    # Case 3: (1, 1, 32, 32) aligned
    q3, k3, v3 = _bhld(1, 1, 32, 32)
    # Case 4: (4, 4, 63, 63) non-aligned
    q4, k4, v4 = _bhld(4, 4, 63, 63)
    # Case 5: (4, 4, 64, 64) aligned
    q5, k5, v5 = _bhld(4, 4, 64, 64)
    # Case 6: (8, 8, 127, 127) non-aligned
    q6, k6, v6 = _bhld(8, 8, 127, 127)
    # Case 7: (8, 8, 128, 128) aligned
    q7, k7, v7 = _bhld(8, 8, 128, 128)
    # Case 8: (16, 16, 255, 255) non-aligned
    q8, k8, v8 = _bhld(16, 16, 255, 255)
    # Case 9: (16, 16, 256, 256) aligned
    q9, k9, v9 = _bhld(16, 16, 256, 256)
    # Case 10: (32, 8, 511, 64) non-aligned (matches static-style head/seq sweep)
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
