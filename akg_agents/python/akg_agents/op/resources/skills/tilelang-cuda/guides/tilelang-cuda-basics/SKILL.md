---
name: tilelang-cuda-basics
description: "TileLang CUDA 核心概念、内核结构和标准编程模式"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  operator_patterns: "all"
---

# TileLang CUDA 编程基础

## 1. 核心概念

### TileLang 简介
- **定义**: TileLang 是专为高性能 GPU/CPU 内核开发设计的领域特定语言（DSL），采用类似 Python 的语法，底层基于 TVM 编译器
- **特点**: 专注于生产力而不牺牲底层优化能力，提供三层抽象级别

### 编程接口层次
- **Level 1 (硬件无关)**: 编译器自动处理内存层次和硬件特定优化，适合快速原型开发
- **Level 2 (硬件感知 + Tile库)**: 提供预定义 Tile 库操作和模式，适合大多数高性能计算应用
- **Level 3 (硬件感知 + 线程原语)**: 提供线程原语和低级构造的直接访问，适合极致性能优化

### 内核 (Kernel)
- **定义**: 使用 `@tilelang.jit` 装饰的函数，编译后在 GPU 上并行执行
- **结构**: 内部包含 `@T.prim_func` 装饰的主函数，通过 `T.Kernel` 上下文管理器定义并行执行逻辑

### 网格 (Grid) 与线程块
- **网格**: 内核启动时的并行维度配置，使用 `T.ceildiv` 计算块数
- **线程块**: 每个块包含指定数量的线程，通过 `threads` 参数设置
- **块索引**: `T.Kernel` 上下文返回 `(bx, by)` 对应 `blockIdx.x, blockIdx.y`

### 内存层次
- **全局内存 (Global Memory)**: GPU 主内存（HBM），所有线程可访问
- **共享内存 (Shared Memory)**: SM 内共享，通过 `T.alloc_shared` 分配
- **寄存器片段 (Fragment)**: 对应 GPU 寄存器文件，通过 `T.alloc_fragment` 分配
- **本地内存 (Local)**: 线程本地存储，通过 `T.alloc_local` 分配

## 2. 标准内核结构

TileLang 内核的标准结构模式：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        # 1. 定义内核上下文（网格和线程配置）
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 2. 内存分配
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            # 3. 初始化
            T.clear(C_local)
            
            # 4. 计算逻辑（含数据加载和计算）
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # 5. 结果写回
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main
```

## 3. 内核调用约定（out_idx）

TileLang 的 `@tilelang.jit` / `tilelang.compile` 通过 `out_idx` 指定哪些张量属于输出。

### 基本用法

```python
@tilelang.jit(out_idx=[1])
def parallel_elementwise_static(length=256):
    @T.prim_func
    def main(A: T.Tensor((length,), "float32"),
             B: T.Tensor((length,), "float32")):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length):
                B[i] = A[i] + 1.0
    return main

kernel = parallel_elementwise_static()
result = kernel(data)  # ✅ 只传输入 data；TileLang 根据 out_idx 返回输出
```

### out_idx 规则

- `out_idx=[-1]`: 最后一个张量为输出
- `out_idx=[1]`: 第二个张量为输出
- 支持多输出：`out1, out2 = kernel(x, y)`

### ⚠️ 常见错误

```python
# ❌ 错误：额外传输出张量
y = torch.empty_like(x)
kernel(x, y)  # ValueError: Expected 2 inputs, got 3 with 2 inputs and 1 outputs

# ✅ 正确：只传输入，out_idx 自动创建输出
result = kernel(x)
```

**实践建议**：
1. **推荐方式**：保留 `out_idx`，调用时只传输入
2. **手动管理输出**：不设置 `out_idx`，在 `prim_func` 里把输出也声明为参数，保证"定义多少参数就传多少参数"

## 4. 基本编程模式

### 4.1 逐元素操作

```python
@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, threads):
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

**关键概念**：
- **并行映射**：每个线程处理一个或多个元素
- **索引计算**：从线程块索引计算全局索引
- **内存访问**：直接访问全局内存

### 4.2 矩阵乘法（GEMM）

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            T.clear(C_local)
            
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main
```

**关键概念**：
- **共享内存缓存**：使用 `T.alloc_shared` 缓存频繁访问的数据
- **软件流水线**：`T.Pipelined` 重叠内存加载和计算
- **内置矩阵乘法**：`T.gemm` 利用 Tensor Core 加速

### 4.3 矩阵向量乘法（GEMV）

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
            
            # ✅ 正确：使用 T.Parallel 获取线程索引
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

**关键概念**：
- **线程索引**：使用 `T.Parallel()` 获取线程索引（推荐方式）
- **串行循环**：`T.serial()` 用于需要串行执行的操作
- **类型转换**：使用 `.astype()` 在计算时进行精度转换
- **⚠️ 注意**：避免使用 `T.get_thread_binding()`，推荐使用 `T.Parallel()`

## 5. 高级编程模式

### 5.1 宏定义

```python
@T.macro
def Softmax(acc_s, acc_s_cast, scores_max, scores_sum, logsum):
    T.copy(scores_max, scores_max_prev)
    T.fill(scores_max, -T.infinity("float"))
    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
    
    for i in T.Parallel(block_M):
        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
    
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
    
    T.reduce_sum(acc_s, scores_sum, dim=1)
    for i in T.Parallel(block_M):
        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    
    T.copy(acc_s, acc_s_cast)
```

### 5.2 条件执行和边界处理

```python
@tilelang.jit(out_idx=[-1])
def conditional_kernel(N, threads):
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

### 5.3 原子操作和归约

```python
@tilelang.jit(out_idx=[-1])
def atomic_reduction(N, K, BLOCK_N, reduce_threads):
    @T.prim_func
    def main(A: T.Tensor((N, K), "float32"),
             C: T.Tensor((N,), "float32")):
        
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            C_shared = T.alloc_shared((BLOCK_N,), "float32")
            C_accum = T.alloc_local((1,), "float32")
            
            T.clear(C_accum)
            
            for tn in T.Parallel(BLOCK_N):
                for k in T.serial(K):
                    C_accum[0] += A[bn * BLOCK_N + tn, k]
                
                T.atomic_add(C_shared[tn], C_accum[0])
                C[bn * BLOCK_N + tn] = C_shared[tn]
    
    return main
```

## 6. 最佳实践总结

### 编程模式选择
- **简单操作**：使用逐元素操作模式
- **矩阵运算**：使用 GEMM 模式，利用 `T.gemm` 内置原语
- **不规则访问**：使用 GEMV 模式
- **复杂计算**：使用宏定义 `@T.macro` 组织代码

### 性能优化要点
1. **合理选择分块大小**：平衡内存使用和计算效率
2. **使用软件流水线**：`T.Pipelined` 重叠内存操作和计算
3. **并行化数据移动**：利用 `T.Parallel` 优化内存访问
4. **选择合适的线程数**：通常为 128 或 256
5. **利用内置原语**：使用 `T.gemm`、`T.reduce_sum` 等优化原语

### 常见错误避免
1. **内存分配过大**：超出硬件限制
2. **索引计算错误**：导致内存访问越界
3. **数据类型不匹配**：精度损失或性能下降
4. **流水线深度不当**：影响性能
5. **⚠️ 同步使用错误**：条件分支中的同步会导致死锁
6. **⚠️ 线程索引获取错误**：使用 `T.get_thread_binding()` 而非 `T.Parallel()`
7. **⚠️ out_idx 使用错误**：额外传输出张量导致参数数量不匹配
