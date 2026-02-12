---
name: triton-ascend-patterns
description: "Triton Ascend 三大编程模式：向量、归约、矩阵"
level: L3
category: method
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  operator_patterns: "elementwise, reduce, matmul"
structure:
  child_skills:
    - triton-ascend-elementwise
    - triton-ascend-reduce
    - triton-ascend-matmul
---

# Triton Ascend 编程模式

## 3.1 向量操作模式

适用于元素级运算：加法、乘法、激活函数等。

### 标准代码结构

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
```

### 适用算子
- 算术运算: add, mul, sub, div
- 激活函数: relu, sigmoid, tanh, gelu
- 数学函数: exp, log, sqrt, pow

### 关键要点
- 正确：使用一维索引和偏移
- 正确：边界处理用 `mask`
- 正确：简单直接的数据流：加载 → 计算 → 存储

## 3.2 归约模式

适用于求和、最大值、最小值等聚合操作。

### 标准代码结构

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
    if pid == 0:  # 只有第一个块写入结果
        tl.atomic_add(output_ptr, block_sum)
```

### 适用算子
- 基础归约: sum, mean, max, min
- 归一化: softmax, logsoftmax, layernorm, batchnorm
- 统计: variance, std

### 关键要点
- 正确：块内归约：使用 `tl.sum`, `tl.max` 等
- 正确：原子操作：使用 `tl.atomic_add` 等写回全局内存
- 正确：数值稳定性：减去最大值防止溢出（见 triton-ascend-reduce）

## 3.3 矩阵乘法模式

适用于矩阵乘法等多维块计算，使用固定核心数启动。

### 标准代码结构

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 关键:使用固定核心数启动,每个核心处理多个块
    pid = tl.program_id(0)  # 核心ID: 0~num_cores-1
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的2D索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K维度循环
        for k in range(0, K, BLOCK_K):
            # 加载A块
            a_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * K + \
                       (k + tl.arange(0, BLOCK_K))[None, :]
            a_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)

            # 加载B块
            b_offset = (k + tl.arange(0, BLOCK_K))[:, None] * N + \
                       (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
            b_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)

            # 矩阵乘累加
            accumulator += tl.dot(a, b)

        # 存储结果
        c_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * N + \
                   (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N)
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)
```

### Host 侧启动

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        num_cores = 20  # Ascend 910B4有20个AI Core
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

        # 关键:固定核心数启动,grid=(num_cores,)不是(NUM_BLOCKS,)
        matmul_kernel[(num_cores,)](
            a, b, c, M, N, K, num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        return c
```

### 适用算子
- 矩阵运算: matmul, bmm (batch matmul), linear
- 卷积: conv2d, conv3d
- 其他多维计算

### 关键要点
- 正确：**固定核心数启动**: 使用 `grid=(num_cores,)` 而非 `(NUM_BLOCKS,)`
- 正确：**循环处理多块**: 每个核心通过 `for block_idx in range(pid, NUM_BLOCKS, num_cores)` 循环处理多个块
- 正确：**分块计算**: 将大矩阵分成小块，减少内存占用
- 正确：**K维度循环**: 累加多个部分乘积
- 错误：**避免错误**: 不要为每个块启动一个程序

## 模式选择指南

| 算子类型 | 推荐模式 | 关键特征 |
|---------|---------|---------|
| Element-wise | 向量操作模式 | 逐元素独立计算 |
| Reduction | 归约模式 | 需要聚合多个值 |
| MatMul/Conv | 矩阵乘法模式 | 多维块计算，固定核心数 |
| Attention | 归约 + 矩阵乘法 | 组合模式，见 triton-ascend-attention |