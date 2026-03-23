---
name: triton-ascend-examples-torch
description: "PyTorch 框架下 Triton Ascend 内核的集成示例，展示 torch 自定义算子注册、tensor 传入传出、grid 配置等标准写法。当目标框架为 torch 时应导入此示例作为代码结构参考。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  framework: torch
---

# PyTorch + Triton Ascend 示例代码

## 示例列表

### 1. Vector Add（向量加法）
```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    
    tl.store(c_ptr + offsets, c, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        c = torch.empty_like(a)
        n_elements = a.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
        return c
```

### 2. MatMul（矩阵乘法）
**核心代码**:
```python
@triton.jit
def matmul_kernel(output_ptr, x_ptr, y_ptr,
                  A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    aidx = tl.arange(0, A)
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)
    
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    
    X = tl.load(x_ptr + Xidx)
    Y = tl.load(y_ptr + Yidx)
    
    result = tl.dot(X, Y)
    
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, result)
```

### 3. Layer Norm（层归一化）
**核心逻辑**:
```python
# 1. 计算均值
mean = tl.sum(x, axis=0) / n_cols

# 2. 计算方差
x_centered = x - mean
variance = tl.sum(x_centered * x_centered, axis=0) / n_cols

# 3. 归一化
variance = tl.maximum(variance, 0.0)  # 防止负数
rstd = 1.0 / tl.sqrt(variance + eps)
normalized = x_centered * rstd

# 4. 应用 weight 和 bias
output = normalized * weight + bias
```

### 4. Double Kernel（双内核调用）
**示例结构**:
```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        # 第一个 kernel
        intermediate = torch.empty_like(x)
        kernel1[grid](x, intermediate, ...)
        
        # 第二个 kernel
        output = torch.empty_like(x)
        kernel2[grid](intermediate, output, ...)
        
        return output
```