---
name: triton-cuda-elementwise
description: "逐元素算子(element-wise)优化策略，包括 add/mul/relu/sigmoid/tanh/gelu/exp/log 等操作的向量化实现和融合技巧。适用于实现激活函数、逐元素运算、广播操作等向量模式算子的 CUDA 内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "elementwise"
  algorithms: "add, mul, relu, sigmoid, tanh, gelu, exp, log, div, sub, sqrt, pow"
---

# Element-wise 算子优化

> 适用于逐元素独立计算的算子

## 适用算子

**算术运算**: add, mul, div, sub, pow
**激活函数**: relu, sigmoid, tanh（需用 `tl.extra.cuda.libdevice.tanh`）, gelu, silu, swish
**数学函数**: exp, log, sqrt, sin, cos, abs

## 优化策略

### 1. 连续内存访问优化

张量在内存中连续存储时，可用一维指针遍历，避免多维索引开销。

**方案 1: 转连续 + 一维访问（推荐）**

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # 非连续张量转为连续（一次性开销）
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        
        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
        return output_tensor

@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)  # 你的计算逻辑
    tl.store(output_ptr + offsets, result, mask=mask)
```

**优势**:
- `.contiguous()` 一次性开销 vs stride 每次访问都有开销
- 更好的合并访问（coalesced access）
- 编译器优化更容易

**方案 2: 使用 stride 访问（不推荐）**

仅当无法调用 `.contiguous()` 时使用。

### 2. BLOCK_SIZE 选择

- **推荐值**: 256, 512, 1024
- **原则**: 平衡并行度和资源占用
- **GPU 考量**:
  - 更大的 BLOCK_SIZE → 更少的 block 启动开销，但可能降低 occupancy
  - 更小的 BLOCK_SIZE → 更细粒度的并行，但启动开销增加
  - 确保 Grid 大小足够大以充分利用 GPU

### 3. Warp 配置

Element-wise 算子通常使用较少的 warp：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
    restore_value=['output_ptr'],  # 必须：列出所有输出指针参数名
)
@triton.jit
def optimized_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 4. 大 Shape 处理

当输入 shape 很大时，确保有足够的 block 来覆盖所有元素：

```python
@triton.jit
def large_elementwise_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # 每个程序处理多个块（grid stride loop）
    for block_start in range(pid * BLOCK_SIZE, n_elements, num_pids * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 启动：限制 Grid 大小
num_blocks = min(triton.cdiv(n_elements, BLOCK_SIZE), 65535)
grid = (num_blocks,)
large_elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=1024)
```

### 5. 向量化加载

对于简单的 element-wise 算子，可以通过更大的 BLOCK_SIZE 来增加每个线程的工作量，提高计算密度：

```python
@triton.jit
def vectorized_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 更大的 BLOCK_SIZE 允许编译器进行更好的向量化
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0.0)  # ReLU
    tl.store(output_ptr + offsets, result, mask=mask)
```

## 完整示例：ReLU

```python
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        
        output = torch.empty_like(x)
        n_elements = x.numel()
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        relu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
        return output
```

## 完整示例：GELU

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x_cubed)  # sqrt(2/pi) ≈ 0.7978845608
    result = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output
```

## 性能检查清单

- [ ] 是否将输入转为连续内存？
- [ ] BLOCK_SIZE 是否为 2 的幂次？
- [ ] 是否使用了 autotune 搜索最优配置？
- [ ] 对于大 shape，是否使用了 grid stride loop？
- [ ] 内存访问是否合并（coalesced）？

## 常见错误

1. **忘记转连续**: 导致非合并访问，性能下降
2. **BLOCK_SIZE 过小**: 启动开销过大
3. **BLOCK_SIZE 过大**: occupancy 降低
4. **忘记 mask**: 越界访问导致错误
5. **不必要的同步**: element-wise 算子不需要同步
