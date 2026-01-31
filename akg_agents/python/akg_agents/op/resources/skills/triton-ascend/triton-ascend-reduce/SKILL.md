---
name: triton-ascend-reduce
description: "归约算子(reduce)优化策略，包含 softmax、layernorm"
level: L4
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  operator_patterns: "reduce"
  algorithms: "sum, mean, max, min, softmax, layernorm, logsoftmax"
---

# Reduce 算子优化

> 适用于需要聚合多个值的归约操作

## 适用算子

**基础归约**: sum, mean, max, min, prod
**归一化**: softmax, logsoftmax, layernorm, batchnorm
**统计**: variance, std

## 通用归约策略

### 1. 块内归约 + 原子操作

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    block_sum = tl.sum(data, axis=0)
    
    # 原子操作写回全局内存
    tl.atomic_add(output_ptr, block_sum)
```

### 2. 数值稳定性处理

**关键**: 对于涉及 exp 的操作（softmax、logsoftmax），必须减去最大值防止溢出。

```python
# ❌ 错误：直接 exp 可能溢出
scores = tl.math.exp2(x)

# ✅ 正确：减去最大值
max_val = tl.max(x, axis=0)
scores = tl.math.exp2(x - max_val)
```

## 特定算子优化

### Softmax

**标准 Softmax**: `output = exp(x - max(x)) / sum(exp(x - max(x)))`

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 1. 加载数据
    row_ptr = input_ptr + row_start
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 2. 数值稳定化：减去最大值
    max_val = tl.max(x, axis=0)
    x_stable = x - max_val
    
    # 3. 计算 exp
    numerator = tl.math.exp2(x_stable * 1.44269504)  # log2(e) ≈ 1.44269504
    
    # 4. 求和
    denominator = tl.sum(numerator, axis=0)
    
    # 5. 归一化
    output = numerator / denominator
    
    # 6. 存储结果
    output_ptr_row = output_ptr + row_start
    tl.store(output_ptr_row + col_offsets, output, mask=mask)
```

**关键点**:
- ✅ 必须减去最大值（数值稳定性）
- ✅ 使用 `tl.math.exp2` 而非 `tl.exp`（Triton 推荐）
- ✅ 注意 `exp2` 需要乘以 log2(e) = 1.44269504

### LayerNorm

**标准 LayerNorm**: `output = (x - mean(x)) / sqrt(var(x) + eps)`

```python
@triton.jit
def layernorm_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 1. 加载数据
    row_ptr = input_ptr + row_start
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    
    # 2. 计算均值
    mean = tl.sum(x, axis=0) / n_cols
    
    # 3. 计算方差
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    
    # 4. 归一化（注意数值稳定性）
    variance = tl.maximum(variance, 0.0)  # 防止负数
    rstd = 1.0 / tl.sqrt(variance + eps)
    normalized = x_centered * rstd
    
    # 5. 可选：应用 weight 和 bias
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + col_offsets, mask=mask)
        normalized = normalized * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_offsets, mask=mask)
        normalized = normalized + bias
    
    # 6. 存储结果
    output_ptr_row = output_ptr + row_start
    tl.store(output_ptr_row + col_offsets, normalized, mask=mask)
```

**关键点**:
- ✅ 防止方差为负：`tl.maximum(variance, 0.0)`
- ✅ 添加 eps 防止除零
- ✅ 使用 float32 进行中间计算（即使输入是 fp16）

### LogSoftmax

**标准 LogSoftmax**: `output = x - max(x) - log(sum(exp(x - max(x))))`

```python
@triton.jit
def logsoftmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row_ptr = input_ptr + row_start
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 数值稳定化
    max_val = tl.max(x, axis=0)
    x_stable = x - max_val
    
    # 计算 log(sum(exp(x)))
    exp_x = tl.math.exp2(x_stable * 1.44269504)
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp) / 1.44269504  # 转回自然对数
    
    # LogSoftmax
    output = x_stable - log_sum_exp
    
    output_ptr_row = output_ptr + row_start
    tl.store(output_ptr_row + col_offsets, output, mask=mask)
```

## 完整示例：Softmax

```python
import torch
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

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        
        softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE)
        return output
```

## 性能优化建议

### 1. 精度提升
使用 float32 进行中间计算，即使输入是 float16/bfloat16：

```python
# 加载时转换为 float32
x = tl.load(input_ptr + offsets, mask=mask)
x = tl.cast(x, tl.float32)

# 计算...

# 存储前转回原精度
result = tl.cast(result, output_dtype)
tl.store(output_ptr + offsets, result, mask=mask)
```

### 2. 逐行处理

对于 2D 数据，通常逐行处理（沿最后一维 reduce）：
- Grid: `(n_rows,)` 每个程序处理一行
- 好处: 每行独立，易于并行

### 3. BLOCK_SIZE 选择

- **推荐**: `triton.next_power_of_2(n_cols)` 向上取到2的幂次
- **原因**: 对齐到2的幂次，编译器优化更好

## 数值稳定性检查清单

- [ ] Softmax/LogSoftmax 是否减去了最大值？
- [ ] 方差计算是否有 `tl.maximum(var, 0.0)`？
- [ ] 除法是否添加了 eps 防止除零？
- [ ] 是否使用 float32 进行中间累加？
- [ ] exp 是否可能溢出（>88 for fp32）？

## 常见错误

1. **忘记减去最大值**: Softmax 直接 exp 导致溢出
2. **使用错误的 exp**: 使用 `tl.exp` 而非 `tl.math.exp2`
3. **精度不足**: 全程使用 fp16 导致累加误差
4. **除零错误**: 方差或和为零时未添加 eps
5. **方差为负**: 数值误差导致方差略小于零
