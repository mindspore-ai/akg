---
name: triton-ascend-example-softmax
description: "Softmax 归约算子的完整 Triton Ascend 实现示例。展示三阶段归约模式（求 max → 求 sum(exp) → 归一化）、分块累加、标量累加器精度提升等技巧。当生成 reduce 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduce"
---

# Softmax — Triton Ascend 实现示例

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    B: tl.constexpr, N: tl.constexpr,
    stride_xb: tl.constexpr, stride_xn: tl.constexpr,
    stride_yb: tl.constexpr, stride_yn: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    for b in range(pid, B, CORE_NUM):
        # Phase 1: max
        max_val = -float('inf')
        for off in range(0, N, BLOCK_SIZE_N):
            n_off = off + tl.arange(0, BLOCK_SIZE_N)
            mask = n_off < N
            x = tl.load(X_ptr + b * stride_xb + n_off * stride_xn,
                        mask=mask, other=-float('inf'))
            max_val = tl.maximum(max_val, tl.max(x, axis=0))

        # Phase 2: sum(exp(x - max))
        sum_val = 0.0
        for off in range(0, N, BLOCK_SIZE_N):
            n_off = off + tl.arange(0, BLOCK_SIZE_N)
            mask = n_off < N
            x = tl.load(X_ptr + b * stride_xb + n_off * stride_xn,
                        mask=mask, other=0.0)
            exp_x = tl.math.exp(x - max_val)
            sum_val += tl.sum(exp_x, axis=0).to(tl.float32)

        # Phase 3: normalize
        for off in range(0, N, BLOCK_SIZE_N):
            n_off = off + tl.arange(0, BLOCK_SIZE_N)
            mask = n_off < N
            x = tl.load(X_ptr + b * stride_xb + n_off * stride_xn,
                        mask=mask, other=0.0)
            result = tl.math.exp(x - max_val) / sum_val
            tl.store(Y_ptr + b * stride_yb + n_off * stride_yn,
                     result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        B, N = x.shape
        y = torch.empty_like(x)
        BLOCK_SIZE_N = 4096
        grid = (self.VEC_CORE_NUM,)
        softmax_kernel[grid](
            x, y, B, N,
            x.stride(0), x.stride(1), y.stride(0), y.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N, CORE_NUM=self.VEC_CORE_NUM)
        return y
```
