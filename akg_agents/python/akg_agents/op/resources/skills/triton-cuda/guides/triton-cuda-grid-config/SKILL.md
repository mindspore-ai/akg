---
name: triton-cuda-grid-config
description: "Grid/Block 配置策略，包括线程块大小选择、SM 占用率优化和大 shape 算子处理方案。适用于需要确定 CUDA kernel 启动参数、优化 GPU 并行效率、或处理超大规模数据的内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Grid 配置策略

Grid 配置是 Triton Kernel 启动的关键。本文档提供 Triton CUDA 的 Grid 配置策略和大 shape 处理方案。

---

## 1. Grid 设置规范

### 维度格式
- **Grid 必须是 tuple 类型**，最多 3 维
- 支持的格式：`(x,)`, `(x, y)`, `(x, y, z)`

```python
# 正确
grid = (100,)
grid = (100, 200)
grid = (100, 200, 50)

# 错误
grid = 100  # 必须是 tuple
grid = [100, 200]  # 必须是 tuple，不能是 list
```

### 使用 lambda（autotune 场景）

当使用 autotune 时，grid 必须使用 lambda：

```python
# autotune 时必须使用 lambda
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# 非 autotune 时可以直接计算
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
```

---

## 2. 1D Grid 配置

### Element-wise 算子

最常见的配置方式：每个 block 处理 BLOCK_SIZE 个元素。

```python
n_elements = input_tensor.numel()
BLOCK_SIZE = 1024
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
```

### 逐行处理（Reduce 类算子）

每个 block 处理一行或多行：

```python
n_rows, n_cols = x.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)

# 方式 1：每行一个 block
grid = (n_rows,)

# 方式 2：限制并行度（grid stride loop）
num_programs = min(n_rows, 65535)
grid = (num_programs,)
```

---

## 3. 2D Grid 配置

### MatMul 类算子

使用 2D Grid 进行行列双向并行：

```python
BLOCK_M, BLOCK_N = 128, 256
grid_m = triton.cdiv(M, BLOCK_M)
grid_n = triton.cdiv(N, BLOCK_N)

# 方式 1：2D Grid
grid = (grid_m, grid_n)

# 方式 2：1D Grid（更灵活，支持 Grouped Ordering）
grid = (grid_m * grid_n,)
```

### 1D vs 2D Grid

| 特性 | 1D Grid | 2D Grid |
|------|---------|---------|
| 灵活性 | 高（支持 Grouped Ordering） | 低 |
| 代码复杂度 | 需要手动计算 pid_m, pid_n | 直接获取 |
| L2 缓存优化 | 容易实现 | 不易实现 |
| 推荐场景 | MatMul（需要缓存优化） | 简单 2D 算子 |

**推荐**: 对于 MatMul 类算子，使用 1D Grid + Grouped Ordering。

---

## 4. 大 Shape 处理：Grid Stride Loop

### 问题描述

CUDA GPU 对 grid 大小也有限制（通常 2^31 - 1 per dimension），但更重要的是，过大的 grid 会导致：
- 启动开销增加
- 资源浪费（每个 block 只处理少量数据）

### Grid Stride Loop 方案

每个 block 通过循环处理多个数据块：

```python
@triton.jit
def grid_stride_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Grid stride loop
    for block_start in range(pid * BLOCK_SIZE, n_elements, num_pids * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 限制 grid 大小
MAX_GRID = 65535
num_blocks = min(triton.cdiv(n_elements, BLOCK_SIZE), MAX_GRID)
grid = (num_blocks,)
grid_stride_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=1024)
```

### Softmax 的 Grid Stride Loop

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # 每个 block 处理多行
    for row_idx in tl.range(row_start, n_rows, row_step):
        # 处理第 row_idx 行...
        pass

# 限制程序数量
num_programs = min(32, n_rows)
softmax_kernel[(num_programs, 1, 1)](...)
```

---

## 5. Batch 维度处理

### 3D Grid

对于 batch 操作，可以使用第三个维度：

```python
# batch matmul
batch_size = Q.shape[0]
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch_size)

@triton.jit
def batch_kernel(...):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_idx = tl.program_id(2)
```

### Flatten Batch

将 batch 维度展平为 1D：

```python
# 将 (batch, M, N) 展平
x_flat = x.view(-1, N)  # (batch * M, N)
total_rows = batch_size * M
grid = (total_rows,)
```

---

## 6. 最佳实践总结

### Element-wise 算子
1. 1D Grid：`(triton.cdiv(n_elements, BLOCK_SIZE),)`
2. BLOCK_SIZE = 1024
3. 大 shape 使用 grid stride loop

### Reduce 算子
1. 1D Grid：`(n_rows,)` 或 `(min(n_rows, max_programs),)`
2. BLOCK_SIZE = triton.next_power_of_2(n_cols)
3. Grid stride loop 处理多行

### MatMul 算子
1. 1D Grid + Grouped Ordering（推荐）
2. 或 2D Grid（简单场景）
3. 使用 autotune 搜索最优配置

### Attention 算子
1. 1D Grid：按查询位置并行
2. 或 3D Grid：(seq_len / BLOCK, heads, batch)

---

## 7. 常见错误和解决方案

### 错误 1: Grid 参数类型错误
```python
# 错误
grid = 1024  # 必须是 tuple
grid = [1024]  # 必须是 tuple

# 正确
grid = (1024,)
```

### 错误 2: Autotune 时未使用 lambda
```python
# 错误：autotune 时直接计算 grid
grid = (triton.cdiv(M, BLOCK_M),)  # BLOCK_M 未知

# 正确：使用 lambda
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
```

### 错误 3: Grid 过大导致性能下降
```python
# 不推荐：每个元素一个 block
grid = (n_elements,)

# 推荐：合理的 block 大小
grid = (triton.cdiv(n_elements, 1024),)
```

---

## 8. 性能调优建议

### Grid 大小选择
1. **足够大**: 充分利用所有 SM（A100 有 108 个 SM）
2. **不过大**: 避免不必要的启动开销
3. **经验值**: Grid 大小通常为 SM 数量的 2-4 倍以上

### 负载均衡
1. 确保每个 block 处理相近的工作量
2. Grid stride loop 天然负载均衡
3. MatMul 的 Grouped Ordering 需要注意边界组的大小

### 与 Autotune 配合
1. Grid 使用 lambda 引用 autotune 参数
2. 不同的 BLOCK_SIZE 会导致不同的 Grid 大小
3. Autotune 会自动搜索最优组合
