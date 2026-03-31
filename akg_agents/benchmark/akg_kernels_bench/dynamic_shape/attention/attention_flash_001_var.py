import torch
import torch.nn as nn


def _bhld(B, H, L, D, dtype=torch.float16):
    """Match static_shape/attention input distribution."""
    q = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    k = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    v = torch.empty(B, H, L, D, dtype=dtype).normal_(mean=0.5, std=0.1)
    return q, k, v


class Model(nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Same as static_shape/attention/attention_flash_001.py (explicit scale)."""
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=self.sm_scale
        )


def get_inputs_dyn_list():
    """Dynamic (B,H,L,D); same sm_scale as static via get_init_inputs."""
    dtype = torch.float16
    cases = [
        (1, 4, 16, 16),
        (2, 8, 32, 32),
        (4, 16, 64, 64),
        (4, 32, 128, 64),
        (8, 16, 256, 64),
        (8, 32, 512, 64),
        (4, 32, 1024, 64),
    ]
    out = []
    for B, H, L, D in cases:
        q, k, v = _bhld(B, H, L, D, dtype=dtype)
        out.append([q, k, v])
    return out


def get_init_inputs():
    sm_scale = 0.5
    return [sm_scale]
