---
name: triton-syntax
description: "Triton语言语法和编程模式，简化GPU kernel开发"
category: dsl
version: "1.0.0"
license: MIT
---

# Triton编程语言

## 概述

Triton是一种专为GPU编程设计的Python DSL，目标是让GPU编程像NumPy一样简单。

## 核心特性

### 1. Python-like语法

```python
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # 程序ID
    pid = tl.program_id(axis=0)
    
    # 块偏移
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask处理边界
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    output = x + y
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2. 自动优化

Triton编译器自动进行：
- 内存合并优化
- 共享内存管理
- 寄存器分配
- 指令调度

### 3. Block编程模型

```python
# 每个program处理一个block的数据
BLOCK_SIZE = 1024
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
```

## 数据类型

### 基本类型

```python
tl.float32    # 32位浮点
tl.float16    # 半精度浮点
tl.bfloat16   # Brain浮点
tl.int32      # 32位整数
tl.int64      # 64位整数
```

### Tensor类型

```python
# 1D tensor
x = tl.arange(0, BLOCK_SIZE)  # shape: [BLOCK_SIZE]

# 2D tensor
rows = tl.arange(0, BLOCK_M)[:, None]  # shape: [BLOCK_M, 1]
cols = tl.arange(0, BLOCK_N)[None, :]  # shape: [1, BLOCK_N]
```

## 内存操作

### 加载（Load）

```python
# 基本加载
data = tl.load(ptr + offsets)

# 带mask加载（处理边界）
data = tl.load(ptr + offsets, mask=mask, other=0.0)

# 带cache hint
data = tl.load(ptr + offsets, cache_modifier=".ca")  # cache all
```

### 存储（Store）

```python
# 基本存储
tl.store(ptr + offsets, data)

# 带mask存储
tl.store(ptr + offsets, data, mask=mask)
```

## 计算操作

### 算术运算

```python
# 逐元素运算
c = a + b
c = a * b
c = a / b
c = tl.exp(a)
c = tl.log(a)
c = tl.sqrt(a)
```

### 规约操作

```python
# Sum
total = tl.sum(x, axis=0)

# Max
maximum = tl.max(x, axis=0)

# Min
minimum = tl.min(x, axis=0)
```

### 矩阵运算

```python
# Dot product
c = tl.dot(a, b)  # 高度优化的矩阵乘法

# 支持混合精度
c = tl.dot(a.to(tl.float16), b.to(tl.float16), acc=tl.float32)
```

## 完整示例

### 1. 向量加法

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### 2. 矩阵乘法

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Load blocks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Compute
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']), 
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c
```

### 3. Softmax

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # 每个program处理一行
    row_idx = tl.program_id(0)
    
    # 列偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载一行
    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # 计算softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # 存储结果
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8
    
    softmax_kernel[(n_rows,)](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return output
```

## 调优技巧

### 1. Block Size选择

```python
# 使用autotune自动选择最优配置
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def my_kernel(...):
    pass
```

### 2. 混合精度

```python
# 使用低精度计算，高精度累加
a_fp16 = a.to(tl.float16)
b_fp16 = b.to(tl.float16)
c = tl.dot(a_fp16, b_fp16, acc=tl.float32)  # 累加用FP32
```

### 3. Prefetching

```python
# 提前加载下一个块的数据
a_next = tl.load(a_ptrs_next, mask=mask_next, other=0.0)
# ... 计算当前块 ...
a = a_next  # 使用预加载的数据
```

## Triton vs CUDA

| 特性 | Triton | CUDA |
|-----|--------|------|
| 语法 | Python-like | C++-like |
| 内存管理 | 自动 | 手动 |
| 优化 | 编译器自动 | 需要手动 |
| 学习曲线 | 平缓 | 陡峭 |
| 性能上限 | 高（接近手写CUDA） | 最高 |
| 可移植性 | 好（支持AMD） | 差（NVIDIA专用） |

## 调试技巧

### 1. 打印调试

```python
@triton.jit
def debug_kernel(...):
    pid = tl.program_id(0)
    if pid == 0:
        tl.device_print("Debug:", value)
```

### 2. 查看生成的PTX

```python
# 编译并查看PTX代码
compiled = triton.compile(my_kernel)
print(compiled.asm['ptx'])
```

### 3. Profile

```python
import triton.profiler as profiler

with profiler.profile([my_kernel]):
    result = my_kernel[grid](*args)
profiler.print_stats()
```

## 常见陷阱

1. **忘记mask**: 边界处理必须使用mask
2. **Block size太小**: 性能不佳
3. **过度使用全局内存**: 应该用block-level计算
4. **数据类型不匹配**: 注意FP16/FP32转换

## 性能优化清单

- [ ] 使用`@triton.autotune`选择最优配置
- [ ] Block size是2的幂
- [ ] 使用`tl.dot`而非手写循环
- [ ] 合理使用mixed precision
- [ ] 检查memory access pattern
- [ ] 使用`num_warps`调优并行度

## 参考资料

- [Triton官方文档](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton论文](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

