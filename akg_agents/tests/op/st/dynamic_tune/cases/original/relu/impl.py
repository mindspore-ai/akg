from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def relu_kernel(X_ptr, Y_ptr, M, BLOCK_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(Y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        M = int(X.shape[0])
        BLOCK_M = 256
        Y = torch.empty_like(X)
        relu_kernel[(triton.cdiv(M, BLOCK_M),)](X, Y, M, BLOCK_M=BLOCK_M)
        return Y
