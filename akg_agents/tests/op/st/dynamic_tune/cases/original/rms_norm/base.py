from __future__ import annotations

import torch  # type: ignore
import torch_npu  # type: ignore
import torch.nn as nn  # type: ignore

DEFAULT_DTYPE = torch.float32
DEFAULT_EPS = 1e-6
HIDDEN = 4096


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN, dtype=DEFAULT_DTYPE))

    def forward(self, X):
        return torch_npu.npu_rms_norm(X, self.weight, epsilon=DEFAULT_EPS)[0]


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
