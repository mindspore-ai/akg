from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

DEFAULT_BIAS = 0.5


@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, bias, M, BLOCK_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x + bias, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def _select_config(self, shape_key):
        from akg_agents.op.dynamic_tune import load_deployed_selector

        selector = load_deployed_selector()
        return selector.select_config(shape_key)

    def forward(self, X, config=None):
        M = int(X.shape[0])
        if config is None:
            shape_key = (int(X.shape[0]),)
            config = self._select_config(shape_key)
        BLOCK_M = config.param("BLOCK_M")
        Y = torch.empty_like(X)
        vector_add_kernel[(triton.cdiv(M, BLOCK_M),)](X, Y, DEFAULT_BIAS, M, BLOCK_M=BLOCK_M)
        return Y
