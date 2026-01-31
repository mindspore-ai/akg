---
name: triton-ascend-examples-torch
description: "PyTorch + Triton Ascend 完整示例代码"
level: L5
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  framework: torch
  examples: "vector_add, matmul, layer_norm, softmax, double_kernel"
---

# PyTorch + Triton Ascend 示例代码

本 Skill 包含完整的可运行示例代码，展示如何在 PyTorch 中使用 Triton Ascend 编写高性能 kernel。

## 示例列表

### 1. Vector Add（向量加法）
**文件**: `torch_vector_add.py` (需补充)
**算子类型**: Element-wise
**关键点**:
- 最简单的 Triton kernel 示例
- 一维索引和 mask
- 标准五步模式

**示例代码结构**:
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
**文件**: `torch_matmul.py`
**算子类型**: MatMul
**关键点**:
- 使用 `tl.dot` 进行矩阵乘法
- 2D 索引计算
- 简单的分块策略

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
**文件**: `torch_layer_norm.py`
**算子类型**: Reduce + Element-wise
**关键点**:
- 逐行处理
- 均值和方差计算
- 数值稳定性处理

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

### 4. Softmax（需补充）
**文件**: `torch_softmax.py` (需补充)
**算子类型**: Reduce
**关键点**:
- 数值稳定化（减去最大值）
- 逐行处理
- exp 和 sum 操作

**标准实现**:
```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # 数值稳定化
    max_val = tl.max(x, axis=0)
    x_stable = x - max_val
    numerator = tl.math.exp2(x_stable * 1.44269504)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)
```

### 5. Double Kernel（双内核调用）
**文件**: `torch_double_kernel.py`
**算子类型**: 多 Kernel 组合
**关键点**:
- 展示如何在一个 forward 中调用多个 kernel
- 中间结果处理
- Kernel 之间的数据传递

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

## 通用模式

所有示例都遵循相同的结构：

### Kernel 定义
```python
@triton.jit
def kernel_name(
    # 输入/输出指针
    output_ptr, input_ptr,
    # 形状参数
    M, N, K,
    # 编译时常量
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取程序 ID
    pid = tl.program_id(0)
    
    # 2. 计算偏移和 mask
    offsets = ...
    mask = ...
    
    # 3. 加载数据
    data = tl.load(...)
    
    # 4. 执行计算
    result = compute(data)
    
    # 5. 存储结果
    tl.store(...)
```

### ModelNew 类
```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 可选：初始化参数、获取核心数等
    
    def forward(self, *inputs):
        # 1. 获取形状参数
        M, N = inputs[0].shape
        
        # 2. 分配输出张量
        output = torch.empty_like(inputs[0])
        
        # 3. 配置 Grid
        grid = (triton.cdiv(M, BLOCK_SIZE),)
        
        # 4. 启动 kernel
        kernel_name[grid](
            output, inputs[0],
            M, N,
            BLOCK_SIZE=1024,
        )
        
        return output
```

## 关键注意事项

### 1. 张量设备和数据类型
```python
# 确保输出张量与输入在同一设备
output = torch.empty_like(input_tensor)  # 推荐
# 或
output = torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
```

### 2. Grid 配置
```python
# 简单情况：直接计算
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

# 2D 情况
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# 大 shape 情况：使用固定核心数
grid = (self.VEC_CORE_NUM,)  # 在 kernel 内循环
```

### 3. ModelNew 格式要求
- **必须**继承 `torch.nn.Module`
- **必须**实现 `forward` 方法
- **不要**使用函数形式（旧版本兼容）

### 4. 参数传递
```python
# 正确：正确：所有参数作为位置参数传递
kernel[grid](output, input, M, N, BLOCK_SIZE=1024)

# 错误：错误：使用关键字参数传递非 constexpr 参数
kernel[grid](output=output, input=input, M=M, N=N)
```

## 使用指南

### 运行示例
```bash
cd /path/to/akg_agents
python python/akg_agents/op/resources/docs/triton_ascend_docs/examples/torch_matmul.py
```

### 修改示例
1. 复制示例代码
2. 根据需求修改 kernel 逻辑
3. 调整 BLOCK_SIZE 和 Grid 配置
4. 测试正确性和性能

### 验证正确性
```python
# 与 PyTorch 原生实现对比
x = torch.randn(128, 256, device='npu', dtype=torch.float16)
output_triton = model_new(x)
output_torch = torch.nn.functional.softmax(x, dim=-1)  # 或其他原生实现

# 检查差异
diff = (output_triton - output_torch).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-3, "Results mismatch!"
```
