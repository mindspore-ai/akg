from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

DEFAULT_DTYPE = torch.float16
K = 4096


class Model(nn.Module):
    def forward(self, A, B):
        return A.float() @ B.float()


def get_init_inputs():
    return []


def get_inputs_dyn_list():
    # 覆盖 M×N 四个象限：小M小N / 小M大N / 大M小N / 大M大N，加上中间档
    # K 固定 4096

    # 象限: 小M 小N
    A0 = torch.randn(1, K, dtype=DEFAULT_DTYPE)
    B0 = torch.randn(K, 1024, dtype=DEFAULT_DTYPE)

    # 象限: 小M 大N
    A1 = torch.randn(1, K, dtype=DEFAULT_DTYPE)
    B1 = torch.randn(K, 8192, dtype=DEFAULT_DTYPE)

    # 中间档
    A2 = torch.randn(16, K, dtype=DEFAULT_DTYPE)
    B2 = torch.randn(K, 4096, dtype=DEFAULT_DTYPE)

    A3 = torch.randn(256, K, dtype=DEFAULT_DTYPE)
    B3 = torch.randn(K, 4096, dtype=DEFAULT_DTYPE)

    A4 = torch.randn(1024, K, dtype=DEFAULT_DTYPE)
    B4 = torch.randn(K, 4096, dtype=DEFAULT_DTYPE)

    # 象限: 大M 小N
    A5 = torch.randn(4096, K, dtype=DEFAULT_DTYPE)
    B5 = torch.randn(K, 1024, dtype=DEFAULT_DTYPE)

    # 象限: 大M 大N
    A6 = torch.randn(4096, K, dtype=DEFAULT_DTYPE)
    B6 = torch.randn(K, 8192, dtype=DEFAULT_DTYPE)

    return [
        [A0, B0],
        [A1, B1],
        [A2, B2],
        [A3, B3],
        [A4, B4],
        [A5, B5],
        [A6, B6],
    ]
