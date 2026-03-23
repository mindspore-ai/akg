---
name: triton-ascend-example-layernorm
description: "LayerNorm 归约算子的完整 Triton Ascend 实现示例。展示两阶段归约模式（统计量计算 → 归一化输出）、标量累加器、分块遍历等技巧。当生成 reduce/normalize 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduce"
---

# LayerNorm — Triton Ascend 实现示例

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_CORES': 40}),
    ],
    key=['feature_size'],
)
@triton.jit
def layernorm_kernel(
    X_ptr, Y_ptr,
    batch_size: tl.constexpr, feature_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_CORES: tl.constexpr,
):
    core_id = tl.program_id(0)
    for batch_idx in range(core_id, batch_size, NUM_CORES):
        batch_offset = batch_idx * feature_size

        # Phase 1: compute mean & variance
        mean_acc = 0.0
        var_acc = 0.0
        for i in range(0, feature_size, BLOCK_SIZE):
            offsets = batch_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_offset + feature_size
            x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            mean_acc += tl.sum(x, axis=0)
            var_acc += tl.sum(x * x, axis=0)

        mean_val = mean_acc / feature_size
        std_val = tl.sqrt(var_acc / feature_size - mean_val * mean_val + eps)

        # Phase 2: normalize
        for i in range(0, feature_size, BLOCK_SIZE):
            offsets = batch_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_offset + feature_size
            x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            tl.store(Y_ptr + offsets, (x - mean_val) / std_val, mask=mask)


def layernorm_triton_ascend(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    batch_size = shape[0]
    feature_size = 1
    for s in shape[1:]:
        feature_size *= s
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty_like(x)
    grid = lambda meta: (meta['NUM_CORES'],)
    layernorm_kernel[grid](x, y, batch_size, feature_size, 1e-5)
    return y
```
