---
name: triton-ascend-example-relu
description: "ReLU 逐元素算子的完整 Triton Ascend 实现示例。展示向量化逐元素操作的标准模式：1D 分块遍历、mask 边界处理、autotune 配置。当生成 elementwise 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# ReLU — Triton Ascend 实现示例

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_CORES': 20}),
    ],
    key=['B', 'N'],
)
@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, N: tl.constexpr,
    stride_b: tl.constexpr, stride_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_CORES: tl.constexpr,
):
    core_id = tl.program_id(0)
    for b in range(core_id, B, NUM_CORES):
        row_start = b * stride_b
        for block_start in range(0, N, BLOCK_SIZE):
            block_len = min(block_start + BLOCK_SIZE, N) - block_start
            offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
            mask = tl.arange(0, BLOCK_SIZE) < block_len
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.maximum(x, 0.0)
            tl.store(y_ptr + offsets, y, mask=mask)


def relu_triton_ascend(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    B, N = x.shape
    y = torch.empty_like(x)
    grid = lambda meta: (meta['NUM_CORES'],)
    relu_kernel[grid](x, y, B, N, x.stride(0), x.stride(1))
    return y
```
