from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

DEFAULT_EPS = 1e-6
ASCEND_NUM_PROGRAMS = 32


@triton.jit
def rms_norm_kernel(
    X_ptr,
    Y_ptr,
    M,
    K,
    eps,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(M, BLOCK_M)
    for blk in range(pid, num_blocks, NUM_PROGRAMS):
        offs_m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        sq_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            sq_sum += tl.sum(x * x, axis=1)
        rstd = 1.0 / tl.sqrt(sq_sum / K + eps)
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            y = x * rstd[:, None]
            tl.store(y_ptrs, y, mask=m_mask[:, None] & k_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._selector = None

    def _select_config(self, shape_key):
        from akg_agents.op.dynamic_tune import load_deployed_selector

        if self._selector is None:
            self._selector = load_deployed_selector()
        return self._selector.select_config(shape_key)

    def forward(self, X, config=None):
        M, K = int(X.shape[0]), int(X.shape[1])
        if config is None:
            shape_key = (M, K)
            config = self._select_config(shape_key)
        BLOCK_M = config.param("BLOCK_M")
        BLOCK_K = config.param("BLOCK_K")
        Y = torch.empty_like(X)
        rms_norm_kernel[(ASCEND_NUM_PROGRAMS,)](
            X,
            Y,
            M,
            K,
            DEFAULT_EPS,
            X.stride(0),
            X.stride(1),
            Y.stride(0),
            Y.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            NUM_PROGRAMS=ASCEND_NUM_PROGRAMS,
        )
        return Y
