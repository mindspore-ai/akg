---
name: triton-ascend-example-matmul
description: "标准矩阵乘法的完整 Triton Ascend 实现示例。展示 2D 分块(tiling)、K 维循环累加、2D mask 处理、Cube Core 利用等关键模式。当生成 matmul 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# 矩阵乘法 — Triton Ascend 实现示例

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512, 'BLOCK_N': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        bm = block_idx // NUM_BLOCKS_N
        bn = block_idx % NUM_BLOCKS_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a_off_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
            a_off_k = k + tl.arange(0, BLOCK_K)
            a_mask = (a_off_m < M)[:, None] & (a_off_k < K)[None, :]
            a = tl.load(a_ptr + a_off_m[:, None] * stride_am
                        + a_off_k[None, :] * stride_ak,
                        mask=a_mask, other=0.0)

            b_off_k = k + tl.arange(0, BLOCK_K)
            b_off_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)
            b_mask = (b_off_k < K)[:, None] & (b_off_n < N)[None, :]
            b = tl.load(b_ptr + b_off_k[:, None] * stride_bk
                        + b_off_n[None, :] * stride_bn,
                        mask=b_mask, other=0.0)
            acc += tl.dot(a, b)

        c_off_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        c_off_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        c_mask = (c_off_m < M)[:, None] & (c_off_n < N)[None, :]
        tl.store(c_ptr + c_off_m[:, None] * stride_cm
                 + c_off_n[None, :] * stride_cn,
                 acc, mask=c_mask)


def matmul_triton_ascend(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    num_cores = 20
    grid = lambda meta: (num_cores,)
    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), num_cores=num_cores)
    return C
```
