---
name: triton-ascend-examples-mindspore
description: "MindSpore + Triton Ascend 完整示例代码"
level: L5
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  framework: mindspore
  examples: "vector_add, matmul, layer_norm, softmax, double_kernel"
---

# MindSpore + Triton Ascend 示例代码

本 Skill 包含完整的可运行示例代码，展示如何在 MindSpore 中使用 Triton Ascend 编写高性能 kernel。

## MindSpore vs PyTorch 差异

| 特性 | PyTorch | MindSpore |
|------|---------|-----------|
| **基类** | `torch.nn.Module` | `mindspore.nn.Cell` |
| **前向函数** | `forward` | `construct` |
| **张量创建** | `torch.empty` | `mindspore.ops.zeros` 或 numpy |
| **设备** | `device='cuda'/'npu'` | 自动管理或 `context.set_context` |
| **数据类型** | `torch.float16` | `mindspore.float16` |

## 示例列表

### 1. Vector Add（向量加法）
**文件**: `mindspore_vector_add.py`
**算子类型**: Element-wise

**MindSpore 实现**:
```python
import mindspore as ms
from mindspore import nn
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    
    tl.store(c_ptr + offsets, c, mask=mask)

class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, a, b):
        # 注意：使用 numpy 创建输出张量
        import numpy as np
        c = ms.Tensor(np.empty_like(a.asnumpy()), dtype=a.dtype)
        
        n_elements = a.size
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
        return c
```

### 2. MatMul（矩阵乘法）
**文件**: `mindspore_matmul.py`
**算子类型**: MatMul

**关键差异**:
```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, x0, x1):  # 注意：使用 construct 而非 forward
        B, C = x0.shape
        C2, D = x1.shape
        assert C == C2, f"矩阵维度不匹配: {C} != {C2}"
        
        # MindSpore 张量创建
        import numpy as np
        output = ms.Tensor(np.empty((B, D), dtype=np.float32))
        
        matmul_kernel[1, 1, 1](output, x0, x1, 1, B, C, D)
        return output
```

### 3. Layer Norm（层归一化）
**文件**: `mindspore_layer_norm.py`
**算子类型**: Reduce + Element-wise

**MindSpore 特有处理**:
```python
class ModelNew(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        
        # MindSpore 参数初始化
        ms.set_seed(0)  # 注意：使用 ms.set_seed 而非 torch.manual_seed
        self.weight = ms.Parameter(ms.ops.ones(normalized_shape, ms.float32))
        self.bias = ms.Parameter(ms.ops.zeros(normalized_shape, ms.float32))
    
    def construct(self, x):
        M, N = x.shape
        import numpy as np
        output = ms.Tensor(np.empty_like(x.asnumpy()), dtype=x.dtype)
        
        grid = (M,)
        layernorm_kernel[grid](
            x, output, self.weight, self.bias,
            N, self.eps, BLOCK_SIZE=triton.next_power_of_2(N)
        )
        return output
```

### 4. Softmax
**文件**: `mindspore_softmax.py`
**算子类型**: Reduce

**实现要点**:
```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, x):
        n_rows, n_cols = x.shape
        import numpy as np
        output = ms.Tensor(np.empty_like(x.asnumpy()), dtype=x.dtype)
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        
        softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE)
        return output
```

### 5. Double Kernel（双内核调用）
**文件**: `mindspore_double_kernel.py`
**算子类型**: 多 Kernel 组合

**多 kernel 调用**:
```python
class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, x):
        import numpy as np
        
        # 第一个 kernel
        intermediate = ms.Tensor(np.empty_like(x.asnumpy()), dtype=x.dtype)
        kernel1[grid](x, intermediate, ...)
        
        # 第二个 kernel
        output = ms.Tensor(np.empty_like(x.asnumpy()), dtype=x.dtype)
        kernel2[grid](intermediate, output, ...)
        
        return output
```

## MindSpore 专用模式

### 1. 张量创建

**方法 1: 使用 numpy（推荐）**
```python
import numpy as np
output = ms.Tensor(np.empty_like(input.asnumpy()), dtype=input.dtype)
```

**方法 2: 使用 MindSpore ops**
```python
output = ms.ops.zeros(shape, dtype=ms.float32)
# 或
output = ms.ops.ones(shape, dtype=ms.float32)
```

### 2. 参数初始化

对于有可学习参数的算子：
```python
class ModelNew(nn.Cell):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 固定随机种子
        ms.set_seed(0)
        
        # 方法1：直接创建参数
        self.weight = ms.Parameter(
            ms.ops.randn(out_features, in_features, dtype=ms.float32)
        )
        self.bias = ms.Parameter(
            ms.ops.zeros(out_features, dtype=ms.float32)
        )
        
        # 方法2：从 nn 层提取（确保权重一致）
        dense = nn.Dense(in_features, out_features)
        self.weight = ms.Parameter(dense.weight.copy())
        self.bias = ms.Parameter(dense.bias.copy()) if dense.bias is not None else None
```

### 3. 设备管理

```python
# 设置运行设备（通常在脚本开头）
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

# 或
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
```

### 4. 数据类型转换

```python
# MindSpore 数据类型
ms.float16
ms.float32
ms.int32
ms.bool_

# 类型转换
tensor_fp32 = tensor_fp16.astype(ms.float32)
```

## 完整示例：Softmax

```python
import mindspore as ms
from mindspore import nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row_ptr = input_ptr + row_start
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 数值稳定化
    max_val = tl.max(x, axis=0)
    x_stable = x - max_val
    numerator = tl.math.exp2(x_stable * 1.44269504)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    
    output_ptr_row = output_ptr + row_start
    tl.store(output_ptr_row + col_offsets, output, mask=mask)

class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, x):
        n_rows, n_cols = x.shape
        
        # 创建输出张量
        import numpy as np
        output = ms.Tensor(np.empty_like(x.asnumpy()), dtype=x.dtype)
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        
        softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE)
        return output

# 使用示例
if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    
    model = ModelNew()
    x = ms.ops.randn(128, 256, dtype=ms.float32)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum per row: {output.sum(axis=1)}")  # 应该接近 1.0
```

## 通用模板

### Cell 类定义
```python
import mindspore as ms
from mindspore import nn
import triton
import triton.language as tl

@triton.jit
def kernel_name(...):
    # Kernel 实现（与 PyTorch 完全相同）
    pass

class ModelNew(nn.Cell):
    def __init__(self):
        super().__init__()
        # 可选：初始化参数
    
    def construct(self, *inputs):  # 注意：使用 construct
        # 1. 获取形状
        M, N = inputs[0].shape
        
        # 2. 创建输出（使用 numpy）
        import numpy as np
        output = ms.Tensor(np.empty((M, N), dtype=np.float32))
        
        # 3. 配置并启动 kernel
        grid = (M,)
        kernel_name[grid](inputs[0], output, ...)
        
        return output
```

## 调试和测试

### 正确性验证
```python
import mindspore as ms
import numpy as np

# 测试
x = ms.ops.randn(128, 256, dtype=ms.float32)
output_triton = model_new(x)

# 与 MindSpore 原生实现对比
output_ms = ms.ops.softmax(x, axis=-1)

# 计算差异
diff = ms.ops.abs(output_triton - output_ms).max()
print(f"Max difference: {diff.asnumpy()}")

# 验证
assert diff.asnumpy() < 1e-3, "Results mismatch!"
```

### 性能测试
```python
import time
import mindspore as ms

# 预热
for _ in range(10):
    _ = model(x)

# 测试
start = time.time()
for _ in range(100):
    _ = model(x)
elapsed = time.time() - start

print(f"Average time: {elapsed/100*1000:.2f} ms")
```

## 常见问题

### 1. 张量设备问题
```python
# 错误：使用 PyTorch 方式
output = torch.empty_like(x)

# 正确：使用 numpy
import numpy as np
output = ms.Tensor(np.empty_like(x.asnumpy()), dtype=x.dtype)
```

### 2. 参数初始化
```python
# 错误：使用 torch.manual_seed
torch.manual_seed(0)

# 正确：使用 ms.set_seed
ms.set_seed(0)
```

### 3. 前向函数命名
```python
# 错误：使用 forward
def forward(self, x):
    pass

# 正确：使用 construct
def construct(self, x):
    pass
```
