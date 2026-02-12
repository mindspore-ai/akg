---
name: triton-ascend-examples-mindspore
description: "MindSpore 框架下 Triton Ascend 内核的完整集成示例，包括 vector_add、matmul、layer_norm、softmax 等标准算子实现。适用于需要参考 MindSpore 自定义算子注册、Primitive 定义方式的内核代码生成场景"
level: L4
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