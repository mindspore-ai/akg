---
name: triton-ascend-basics
description: "Triton Ascend 编程基础，包括核心概念（program_id、block、grid）、内核函数结构、装饰器用法和标准代码模式。适用使用 Triton Ascend、需要了解基本语法结构的任意内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_patterns: "all"
---

# Triton Ascend 编程基础

## 标准内核结构（交错循环）

```python
@triton.jit
def kernel(
    output_ptr, input_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

## 内核启动模板

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (self.VEC_CORE_NUM,)  # Ascend: 固定为核心数
        kernel[grid](out, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=self.VEC_CORE_NUM)
        return out
```

## 边界处理

```python
offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
result = tl.where(condition, true_val, false_val)
```

## Autotune 用法

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pass
```

### Autotune 关键要点
1. **grid 必须使用 lambda**: `grid = lambda meta: (...)`
2. **调用时不传 configs 参数**: autotune 自动传入
3. **configs 参数必须是 constexpr**
4. **key 参数**: 指定哪些维度变化时重新 autotune
5. **Ascend 不支持调优**: num_warps / num_ctas / num_stages 等参数
