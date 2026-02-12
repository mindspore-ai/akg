---
name: triton-ascend-reduce
description: "归约算子(reduce)优化策略，包含 sum/mean/max/min、softmax、layernorm、logsoftmax 等实现技巧。适用于需要在 Ascend NPU 上实现任意维度归约、规范化层或注意力分数计算的内核代码生成场景"
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
# 错误：错误：直接 exp 可能溢出
scores = tl.math.exp2(x)

# 正确：正确：减去最大值
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
- 正确：必须减去最大值（数值稳定性）
- 正确：使用 `tl.math.exp2` 而非 `tl.exp`（Triton 推荐）
- 正确：注意 `exp2` 需要乘以 log2(e) = 1.44269504

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
- 正确：防止方差为负：`tl.maximum(variance, 0.0)`
- 正确：添加 eps 防止除零
- 正确：使用 float32 进行中间计算（即使输入是 fp16）

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
