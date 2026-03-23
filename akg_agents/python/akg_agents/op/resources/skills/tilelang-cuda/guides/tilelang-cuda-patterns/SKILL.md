---
name: tilelang-cuda-patterns
description: "TileLang CUDA 核心编程模式（逐元素、归约、矩阵乘法、GEMV）的标准实现范式和代码模板。适用于需要快速确定算子属于哪种编程模式、或需要了解 TileLang 各模式基本代码结构的内核代码生成场景"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  operator_patterns: "elementwise, reduce, matmul, gemv"
---

# TileLang CUDA 编程模式

## 1. 逐元素操作模式

适用于元素级运算：加法、乘法、激活函数等。

### 标准代码结构

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def elementwise_op(M, N, block_M, block_N, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"),
             B: T.Tensor((M, N), "float32"),
             C: T.Tensor((M, N), "float32")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for (local_y, local_x) in T.Parallel(block_M, block_N):
                y = by * block_M + local_y
                x = bx * block_N + local_x
                C[y, x] = A[y, x] + B[y, x]
    
    return main
```

### 适用算子
- 算术运算: add, mul, sub, div
- 激活函数: relu, sigmoid, tanh, gelu
- 数学函数: exp, log, sqrt, pow
- 类型转换: cast
- 广播操作: broadcast

### 关键要点
- 使用 `T.Parallel` 映射到线程
- 从块索引计算全局索引
- 直接访问全局内存，无需共享内存
- 适当处理边界条件

### 条件逐元素操作

```python
@tilelang.jit(out_idx=[-1])
def conditional_elementwise(N, threads):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"),
             B: T.Tensor((N,), "float32"),
             C: T.Tensor((N,), "float32")):
        
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < N:
                    C[idx] = T.if_then_else(
                        A[idx] > 0,
                        A[idx] + B[idx],
                        A[idx] - B[idx]
                    )
    
    return main
```

## 2. 归约模式

适用于求和、最大值、归一化等聚合操作。

### 标准代码结构

```python
@tilelang.jit(out_idx=[-1])
def reduction_op(M, N, block_size):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"),
             C: T.Tensor((M,), "float32")):
        
        with T.Kernel(M, threads=block_size) as bx:
            # 分配寄存器片段
            A_local = T.alloc_fragment((N,), "float32")
            C_local = T.alloc_fragment((1,), "float32")
            
            # 加载数据
            T.copy(A[bx, 0:N], A_local)
            
            # ✅ 使用内置归约函数
            T.reduce_sum(A_local, C_local, dim=0)
            
            # 写回结果
            C[bx] = C_local[0]
    
    return main
```

### 适用算子
- 基础归约: sum, mean, max, min, prod
- 归一化: softmax, layernorm, batchnorm, rmsnorm
- 统计: variance, std
- 加权归约: weighted sum

### 关键要点
- **✅ 必须使用内置归约函数**: `T.reduce_sum`, `T.reduce_max`, `T.reduce_min`, `T.reduce_mean`
- **❌ 禁止手动归约**: 手动归约需要 `T.sync_threads()`，在条件分支中极易导致死锁
- 使用寄存器片段 `T.alloc_fragment` 作为临时存储
- 注意数值稳定性（如 softmax 中先减去最大值）

### LayerNorm 归约示例

```python
@tilelang.jit(out_idx=[-1])
def layer_norm(M, N, block_size):
    @T.prim_func
    def main(x: T.Tensor((M, N), "float32"),
             y: T.Tensor((M, N), "float32")):
        
        with T.Kernel(M, threads=block_size) as bx:
            A_local = T.alloc_fragment((N,), "float32")
            A_pow_local = T.alloc_fragment((N,), "float32")
            A_sum = T.alloc_fragment((1,), "float32")
            A_powsum = T.alloc_fragment((1,), "float32")
            
            # 加载数据
            for tid in T.Parallel(N):
                A_local[tid] = x[bx, tid]
                A_pow_local[tid] = x[bx, tid] * x[bx, tid]
            
            # ✅ 使用内置归约
            T.reduce_sum(A_local, A_sum, dim=0)
            T.reduce_sum(A_pow_local, A_powsum, dim=0)
            
            # 计算均值和方差
            for tid in T.Parallel(N):
                mean_val = A_sum[0] / N
                var_val = A_powsum[0] / N - mean_val * mean_val
                y[bx, tid] = (A_local[tid] - mean_val) / T.sqrt(var_val + 1e-5)
    
    return main
```

## 3. 矩阵乘法模式

适用于矩阵乘法等多维块计算。

### 标准代码结构

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 共享内存分配
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            T.clear(C_local)
            
            # K 维度循环（软件流水线）
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # 结果写回
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main
```

### 适用算子
- 矩阵运算: matmul, bmm (batch matmul), linear
- 卷积: conv2d, conv3d
- 注意力机制: attention (Q*K^T, scores*V)

### 关键要点
- **2D Grid**: 使用 `T.Kernel(grid_x, grid_y)` 二维并行
- **共享内存**: 使用 `T.alloc_shared` 缓存数据块
- **K 维度循环**: 使用 `T.Pipelined` 进行软件流水线
- **内置 GEMM**: 使用 `T.gemm` 利用 Tensor Core 加速
- **混合精度**: 输入用 float16，累加器用 float32

## 4. GEMV 模式

适用于矩阵向量乘法等不规则访问模式。

### 标准代码结构

```python
@tilelang.jit(out_idx=[-1])
def gemv(N, K, BLOCK_N, BLOCK_K):
    @T.prim_func
    def main(A: T.Tensor((K,), "float16"),
             B: T.Tensor((N, K), "float16"),
             C: T.Tensor((N,), "float16")):
        
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            A_shared = T.alloc_shared((BLOCK_K,), "float16")
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), "float16")
            
            for tn in T.Parallel(BLOCK_N):
                C_reg = T.alloc_local((1,), "float")
                T.clear(C_reg)
                
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for tk in T.serial(BLOCK_K):
                        A_shared[tk] = A[bk * BLOCK_K + tk]
                        B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                    
                    for tk in T.serial(BLOCK_K):
                        C_reg[0] += A_shared[tk].astype("float") * B_shared[tn, tk].astype("float")
                
                C[bn * BLOCK_N + tn] = C_reg[0]
    
    return main
```

### 关键要点
- **线程索引**: 使用 `T.Parallel` 获取线程索引
- **串行循环**: 使用 `T.serial` 进行 K 维度遍历
- **本地寄存器**: 使用 `T.alloc_local` 存储线程私有累加值
- **类型转换**: 在计算时进行精度转换 `.astype("float")`

## 模式选择指南

| 算子类型 | 推荐模式 | 关键特征 | 内存使用 |
|---------|---------|---------|---------|
| Element-wise | 逐元素操作 | 逐元素独立计算 | 全局内存 |
| Reduction | 归约模式 | 需要聚合多个值 | 寄存器片段 |
| MatMul/Conv | 矩阵乘法模式 | 多维块计算，2D Grid | 共享内存 + 寄存器 |
| GEMV | GEMV 模式 | 向量-矩阵乘法 | 共享内存 + 本地内存 |
| Attention | 归约 + 矩阵乘法 | 组合模式 | 共享内存 + 寄存器 |

## 最佳实践

1. **选择合适的模式**: 根据算子特性选择基础模式
2. **优化分块大小**: 平衡并行度和资源占用
3. **使用内置原语**: 优先使用 `T.gemm`, `T.reduce_*` 等
4. **注意数值稳定性**: 对于 reduce 类算子特别注意
5. **内存访问优化**: 使用 `T.copy` 进行合并访问
6. **利用软件流水线**: 使用 `T.Pipelined` 隐藏内存延迟
7. **避免手动同步**: 使用内置归约函数替代手动归约
