# TileLang 基础编程指南

## 概述

本文档提供 TileLang 的基础编程模式和实际示例，帮助开发者快速上手并掌握 TileLang 的核心编程概念。

## 基本编程模式

### 1. 矩阵乘法（GEMM）

矩阵乘法是 GPU 计算中最基础也是最重要的操作之一。以下是 TileLang 中实现 GEMM 的完整示例：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), 
             B: T.Tensor((K, N), "float16"), 
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 内存分配
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            T.clear(C_local)
            
            # K 维度循环
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # 数据加载
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                
                # 计算
                T.gemm(A_shared, B_shared, C_local)
            
            # 结果写回
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main

```

### 2. 逐元素操作

逐元素操作是最简单的并行操作，适合初学者理解 TileLang 的基本概念：

```python
@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"),
             B: T.Tensor((M, N), "float32"),
             C: T.Tensor((M, N), "float32")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # 并行计算
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

### 3. 矩阵向量乘法（GEMV）

矩阵向量乘法展示了如何处理不规则的数据访问模式：

```python
@tilelang.jit(out_idx=[-1])
def gemv(N, K, BLOCK_N, BLOCK_K):
    @T.prim_func
    def main(A: T.Tensor((K,), "float16"),
             B: T.Tensor((N, K), "float16"),
             C: T.Tensor((N,), "float16")):
        
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            # 内存分配
            A_shared = T.alloc_shared((BLOCK_K,), "float16")
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), "float16")
            
            # ✅ 正确：使用 T.Parallel 获取线程索引
            for tn in T.Parallel(BLOCK_N):
                # tn 对应 threadIdx.x
                C_reg = T.alloc_local((1,), "float")
                T.clear(C_reg)
                
                # K 维度循环
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    # 数据加载
                    for tk in T.serial(BLOCK_K):
                        A_shared[tk] = A[bk * BLOCK_K + tk]
                        B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                    
                    # 计算
                    for tk in T.serial(BLOCK_K):
                        C_reg[0] += A_shared[tk].astype("float") * B_shared[tn, tk].astype("float")
                
                C[bn * BLOCK_N + tn] = C_reg[0]
    
    return main

```

**关键概念**：
- **线程索引**：使用 `T.Parallel()` 获取线程索引（推荐方式）
- **串行循环**：某些操作需要串行执行
- **类型转换**：在计算时进行精度转换
- **⚠️ 注意**：避免使用 `T.get_thread_binding()`，推荐使用 `T.Parallel()`


## 高级编程模式

### 1. 使用宏定义复杂操作

```python
@T.macro
def Softmax(acc_s, acc_s_cast, scores_max, scores_sum, logsum):
    """Softmax 宏定义"""
    # 保存之前的最大值
    T.copy(scores_max, scores_max_prev)
    
    # 计算新的最大值
    T.fill(scores_max, -T.infinity("float"))
    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
    
    # 计算缩放因子
    for i in T.Parallel(block_M):
        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
    
    # 应用 softmax
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
    
    # 计算归一化因子
    T.reduce_sum(acc_s, scores_sum, dim=1)
    for i in T.Parallel(block_M):
        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    
    T.copy(acc_s, acc_s_cast)

# 在 Flash Attention 中使用
def flash_attention_with_macro():
    # ... 其他代码 ...
    
    for k in T.Pipelined(loop_range, num_stages=num_stages):
        # 计算注意力分数
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)
        
        # 使用宏进行 softmax
        Softmax(acc_s, acc_s_cast, scores_max, scores_sum, logsum)
        
        # 计算输出
        T.gemm(acc_s_cast, V_shared, acc_o)
```

### 2. 条件执行和边界处理

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
                
                # 边界检查
                if idx < N:
                    # 条件执行
                    C[idx] = T.if_then_else(
                        A[idx] > 0,
                        A[idx] + B[idx],  # 正数时相加
                        A[idx] - B[idx]   # 负数时相减
                    )
    
    return main
```

### 3. 原子操作和归约

```python
@tilelang.jit(out_idx=[-1])
def atomic_reduction(N, K, BLOCK_N, reduce_threads):
    @T.prim_func
    def main(A: T.Tensor((N, K), "float32"),
             C: T.Tensor((N,), "float32")):
        
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            
            # 共享内存用于归约
            C_shared = T.alloc_shared((BLOCK_N,), "float32")
            C_accum = T.alloc_local((1,), "float32")
            
            # 初始化共享内存
            if tk == 0:
                C_shared[tn] = 0
            
            T.clear(C_accum)
            
            # 计算部分和
            for k in T.serial(K):
                C_accum[0] += A[bn * BLOCK_N + tn, k]
            
            # 原子归约
            T.atomic_add(C_shared[tn], C_accum[0])
            
            # ⚠️ 注意：如果需要同步，确保所有线程都参与
            # T.sync_threads()  # 所有线程都必须执行此操作
            
            # 写回结果
            C[bn * BLOCK_N + tn] = C_shared[tn]
    
    return main
```

## 最佳实践总结

### 1. 编程模式选择

- **简单操作**：使用逐元素操作模式
- **矩阵运算**：使用 GEMM 模式
- **不规则访问**：使用 GEMV 模式
- **复杂计算**：使用宏定义模式

### 2. 性能优化要点

1. **合理选择分块大小**：平衡内存使用和计算效率
2. **使用软件流水线**：重叠内存操作和计算
3. **并行化数据移动**：利用 `T.Parallel` 优化内存访问
4. **选择合适的线程数**：通常为 128 或 256
5. **利用内置原语**：使用 `T.gemm` 等优化原语

### 3. 常见错误避免

1. **内存分配过大**：超出硬件限制
2. **索引计算错误**：导致内存访问越界
3. **数据类型不匹配**：精度损失或性能下降
4. **流水线深度不当**：影响性能
5. **⚠️ 同步使用错误**：条件分支中的同步会导致死锁
6. **⚠️ 线程索引获取错误**：使用 `T.get_thread_binding()` 而非 `T.Parallel()`


