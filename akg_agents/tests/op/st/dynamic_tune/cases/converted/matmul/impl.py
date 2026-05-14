from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

from akg_agents.op.dynamic_tune import load_deployed_selector

ASCEND_CUBE_CORE_NUM = 20


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    CORE_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        bm = block_idx // NUM_BLOCKS_N
        bn = block_idx % NUM_BLOCKS_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_off_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
            a_off_k = k + tl.arange(0, BLOCK_K)
            a_mask = (a_off_m < M)[:, None] & (a_off_k < K)[None, :]
            a = tl.load(a_ptr + a_off_m[:, None] * stride_am + a_off_k[None, :] * stride_ak, mask=a_mask, other=0.0)
            b_off_k = k + tl.arange(0, BLOCK_K)
            b_off_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)
            b_mask = (b_off_k < K)[:, None] & (b_off_n < N)[None, :]
            b = tl.load(b_ptr + b_off_k[:, None] * stride_bk + b_off_n[None, :] * stride_bn, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)
        c_off_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        c_off_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        c_mask = (c_off_m < M)[:, None] & (c_off_n < N)[None, :]
        tl.store(c_ptr + c_off_m[:, None] * stride_cm + c_off_n[None, :] * stride_cn, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def _select_config(self, shape_key):
        selector = load_deployed_selector()
        return selector.select_config(shape_key)

    def forward(self, A, B, config=None):
        M, K = int(A.shape[0]), int(A.shape[1])
        N = int(B.shape[1])
        shape_key = (int(A.shape[0]), int(B.shape[1]), int(A.shape[1]))
        if config is None:
            config = self._select_config(shape_key)
        BLOCK_M = config.param("BLOCK_M")
        BLOCK_N = config.param("BLOCK_N")
        BLOCK_K = config.param("BLOCK_K")
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        matmul_kernel[(ASCEND_CUBE_CORE_NUM,)](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            CORE_NUM=ASCEND_CUBE_CORE_NUM,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
        )
        return C
