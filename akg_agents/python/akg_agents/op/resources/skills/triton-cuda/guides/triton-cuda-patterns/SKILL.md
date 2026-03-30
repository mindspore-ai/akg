---
name: triton-cuda-patterns
description: "Triton CUDA 三大核心编程模式（向量/逐元素、归约、矩阵乘法）的标准实现范式和代码模板。适用于需要快速确定算子属于哪种编程模式、或需要了解各模式基本代码结构的 CUDA 内核代码生成场景"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "elementwise, reduce, matmul"
structure:
  child_skills:
    - triton-cuda-elementwise
    - triton-cuda-reduce
    - triton-cuda-matmul
---

# Triton CUDA 编程模式

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
- 激活函数: relu, sigmoid, tanh（需用 `tl.extra.cuda.libdevice.tanh`）, gelu
- 数学函数: exp, log, sqrt, pow

### 关键要点
- 使用一维索引和偏移
- 边界处理用 `mask`
- 简单直接的数据流：加载 → 计算 → 存储

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
    tl.atomic_add(output_ptr, block_sum)
```

### 适用算子
- 基础归约: sum, mean, max, min
- 归一化: softmax, logsoftmax, layernorm, batchnorm
- 统计: variance, std

### 关键要点
- 块内归约：使用 `tl.sum`, `tl.max` 等
- 原子操作：使用 `tl.atomic_add` 等写回全局内存
- 数值稳定性：减去最大值防止溢出（见 triton-cuda-reduce）

## 3.3 矩阵乘法模式

适用于矩阵乘法等多维块计算。

### 标准代码结构

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 获取程序 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K 维度循环
    for k in range(0, K, BLOCK_SIZE_K):
        # 创建块指针
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )

        # 加载数据块
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # 矩阵乘累加
        accumulator += tl.dot(a, b)

    # 存储结果（需显式转换类型，匹配输出 dtype）
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
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

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
        )
        return c
```

### 适用算子
- 矩阵运算: matmul, bmm (batch matmul), linear
- 卷积: conv2d, conv3d（im2col 变换后）
- 其他多维计算

### 关键要点
- **2D Grid**: 使用 `grid=(grid_m, grid_n)` 二维并行
- **分块计算**: 将大矩阵分成小块，减少内存占用
- **K 维度循环**: 累加多个部分乘积
- **block_ptr**: 使用 `tl.make_block_ptr` 简化 2D 数据访问
- **Tensor Core**: 使用 `tl.dot` 自动利用 Tensor Core

## 模式选择指南

| 算子类型 | 推荐模式 | 关键特征 |
|---------|---------|---------|
| Element-wise | 向量操作模式 | 逐元素独立计算 |
| Reduction | 归约模式 | 需要聚合多个值 |
| MatMul/Conv | 矩阵乘法模式 | 多维块计算，2D Grid |
| Attention | 归约 + 矩阵乘法 | 组合模式，见 triton-cuda-attention |

## 最佳实践

1. **选择合适的模式**: 根据算子特性选择基础模式
2. **优化块大小**: 平衡并行度和资源占用
3. **注意边界**: 使用 mask 处理不规则形状
4. **数值稳定性**: 对于 reduce 类算子特别注意
5. **内存访问**: 优化数据布局，提高缓存命中率
6. **利用硬件特性**: 使用 num_warps/num_stages 优化流水线
