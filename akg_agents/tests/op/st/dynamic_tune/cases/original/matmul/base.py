from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

DEFAULT_DTYPE = torch.float32
K = 4096


class Model(nn.Module):
    def forward(self, A, B):
        return A @ B


def get_init_inputs():
    return []


def get_inputs_dyn_list():
    # M × N 各取小中大: 3 × 3 = 9 cases，K 固定 4096
    # M: 1, 128, 4096  N: 128, 4096, 8192
    shapes = [
        (1, 128), (1, 4096), (1, 8192),
        (128, 128), (128, 4096), (128, 8192),
        (4096, 128), (4096, 4096), (4096, 8192),
    ]
    return [[torch.randn(m, K, dtype=DEFAULT_DTYPE), torch.randn(K, n, dtype=DEFAULT_DTYPE)] for m, n in shapes]
