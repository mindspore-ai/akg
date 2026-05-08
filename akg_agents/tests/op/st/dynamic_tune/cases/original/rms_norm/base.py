from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

DEFAULT_DTYPE = torch.float32
DEFAULT_EPS = 1e-6
HIDDEN = 4096


class Model(nn.Module):
    def forward(self, X):
        K = int(X.shape[-1])
        rstd = 1.0 / torch.sqrt(X.float().pow(2).sum(dim=-1, keepdim=True) / K + DEFAULT_EPS)
        return (X.float() * rstd).to(X.dtype)


def get_init_inputs():
    return []


def get_inputs_dyn_list():
    inputs1 = torch.randn(1, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs2 = torch.randn(4, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs3 = torch.randn(16, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs4 = torch.randn(32, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs5 = torch.randn(64, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs6 = torch.randn(128, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs7 = torch.randn(256, HIDDEN, dtype=DEFAULT_DTYPE)
    inputs8 = torch.randn(512, HIDDEN, dtype=DEFAULT_DTYPE)
    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
        [inputs5],
        [inputs6],
        [inputs7],
        [inputs8],
    ]
