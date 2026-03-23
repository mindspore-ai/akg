---
name: triton-cuda-reduce
description: "归约算子(reduce)优化策略，包含 sum/mean/max/min、softmax、layernorm、logsoftmax 等实现技巧。适用于需要在 CUDA GPU 上实现任意维度归约、规范化层或注意力分数计算的内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
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
# 错误：直接 exp 可能溢出
exp_val = tl.exp(x)

# 正确：减去最大值
max_val = tl.max(x, axis=0)
exp_val = tl.exp(x - max_val)
```

## 特定算子优化

### Softmax

**标准 Softmax**: `output = exp(x - max(x)) / sum(exp(x - max(x)))`

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # 获取当前程序处理的行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        # 计算当前行的起始指针
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # 创建列偏移
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # 加载数据，使用掩码处理边界
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # 数值稳定性：减去最大值
        row_minus_max = row - tl.max(row, axis=0)

        # 计算指数（CUDA 后端直接使用 tl.exp）
        numerator = tl.exp(row_minus_max)

        # 计算分母（归一化因子）
        denominator = tl.sum(numerator, axis=0)

        # 计算 softmax
        softmax_output = numerator / denominator

        # 存储结果
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

**关键点**:
- 必须减去最大值（数值稳定性）
- CUDA 后端直接使用 `tl.exp`（不需要像 Ascend 那样使用 `tl.math.exp2`）
- 使用 `tl.range` 实现多行处理（grid stride loop）

### LayerNorm

**标准 LayerNorm**: `output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`

```python
@triton.jit
def layer_norm_kernel(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 获取当前程序处理的行
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # 第一遍：计算均值
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # 第二遍：计算方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 保存 mean 和 rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # 第三遍：归一化并应用线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
```

**关键点**:
- 多遍扫描：分别计算均值、方差、归一化（适用于 N > BLOCK_SIZE 的情况）
- 使用 float32 进行中间计算（即使输入是 fp16）
- 保存 mean 和 rstd 供反向传播使用

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
    exp_x = tl.exp(x_stable)
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp)
    
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
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)

        num_programs = min(32, n_rows)  # 限制程序数量

        softmax_kernel[(num_programs, 1, 1)](
            y, x,
            x.stride(0), y.stride(0),
            n_rows, n_cols,
            BLOCK_SIZE
        )
        return y
```

## 性能优化建议

### 1. 精度提升
使用 float32 进行中间计算，即使输入是 float16/bfloat16：

```python
# 加载时转换为 float32
x = tl.load(input_ptr + offsets, mask=mask)
x = x.to(tl.float32)

# 计算...

# 存储前转回原精度
result = result.to(tl.float16)
tl.store(output_ptr + offsets, result, mask=mask)
```

### 2. 逐行处理

对于 2D 数据，通常逐行处理（沿最后一维 reduce）：
- Grid: `(n_rows,)` 或 `(min(n_rows, max_programs),)` 每个程序处理一行或多行
- 好处: 每行独立，易于并行

### 3. BLOCK_SIZE 选择

- **推荐**: `triton.next_power_of_2(n_cols)` 向上取到 2 的幂次
- **原因**: 对齐到 2 的幂次，编译器优化更好
- **大列数**: 当 n_cols 很大时，使用多遍扫描（for 循环）

### 4. Grid Stride Loop

当行数很大时，使用 grid stride loop 而非为每行分配一个 block：

```python
row_start = tl.program_id(0)
row_step = tl.num_programs(0)
for row_idx in tl.range(row_start, n_rows, row_step):
    # 处理第 row_idx 行
```

## 数值稳定性检查清单

- [ ] Softmax/LogSoftmax 是否减去了最大值？
- [ ] 方差计算是否有 `tl.maximum(var, 0.0)` 或 eps 保护？
- [ ] 除法是否添加了 eps 防止除零？
- [ ] 是否使用 float32 进行中间累加？
- [ ] exp 是否可能溢出（>88 for fp32）？

## 常见错误

1. **忘记减去最大值**: Softmax 直接 exp 导致溢出
2. **精度不足**: 全程使用 fp16 导致累加误差
3. **除零错误**: 方差或和为零时未添加 eps
4. **方差为负**: 数值误差导致方差略小于零
5. **边界处理**: other 参数设置不当（softmax 应使用 `-inf`，sum 应使用 `0.0`）
