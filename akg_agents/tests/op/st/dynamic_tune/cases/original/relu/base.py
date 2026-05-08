from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

DEFAULT_DTYPE = torch.float32


class Model(nn.Module):
    def forward(self, X):
        return torch.relu(X)


def get_init_inputs():
    return []


def get_inputs_dyn_list():
    inputs1 = torch.randn(256, dtype=DEFAULT_DTYPE)
    inputs2 = torch.randn(512, dtype=DEFAULT_DTYPE)
    inputs3 = torch.randn(1024, dtype=DEFAULT_DTYPE)
    inputs4 = torch.randn(2048, dtype=DEFAULT_DTYPE)
    inputs5 = torch.randn(4096, dtype=DEFAULT_DTYPE)
    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
        [inputs5],
    ]
