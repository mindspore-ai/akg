---
name: triton-ascend-elementwise
description: "逐元素算子(element-wise)优化策略，包括 add/mul/relu/sigmoid/tanh/gelu/exp/log 等操作的向量化实现和融合技巧。适用于实现激活函数、逐元素运算、广播操作等向量模式算子的内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
  operator_patterns: "elementwise"
  algorithms: "add, mul, relu, sigmoid, tanh, gelu, exp, log, div, sub, sqrt, pow"
---

# Element-wise 算子优化

> 适用于逐元素独立计算的算子

## 适用算子

**算术运算**: add, mul, div, sub, pow
**激活函数**: relu, sigmoid, tanh, gelu, silu, swish
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
- 正确：`.contiguous()` 一次性开销 vs stride 每次访问都有开销
- 正确：更好的缓存命中率
- 正确：编译器优化更容易

**方案 2: 使用 stride 访问（不推荐）**

仅当无法调用 `.contiguous()` 时使用。

### 2. 核心数配置

Element-wise 算子使用 **VEC核心数**（向量计算核心）。

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取核心数，只执行一次
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40  # Ascend 910B4 默认
```

### 3. BLOCK_SIZE 选择

- **推荐值**: 1024-2048
- **原则**: 平衡并行度和资源占用
- **考虑因素**:
  - 更大的 BLOCK_SIZE → 更少的 Grid 启动开销
  - 更小的 BLOCK_SIZE → 更细粒度的并行

### 4. 核内循环优化

对于简单的 element-wise 算子，可以通过切分并添加核内循环来隐藏搬运计算开销：

```python
@triton.jit
def optimized_elementwise(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # 添加核内循环，让编译器进行流水优化
    for i in range(4):  # 展开4次
        offsets = (pid * 4 + i) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

编译器会自动将核内 for 循环进行多级流水处理。