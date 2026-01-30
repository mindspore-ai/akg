# TileLang 性能优化策略文档

## 概述

本文档提供了 TileLang 性能优化的全面指南，专注于 CUDA 平台的优化策略。通过合理应用这些策略，可以显著提升内核性能。

## 1. 内存优化策略

### 1.1 内存层次结构优化

```python
def memory_optimized_matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"), 
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 1. 共享内存分配 - 用于缓存频繁访问的数据
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            
            # 2. 寄存器片段分配 - 用于累加和临时存储
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            # 3. 启用swizzle以提高 L2 缓存局部性
            T.use_swizzle(panel_size=10, enable=True)
            
            T.clear(C_local)
            
            # 4. 软件流水线优化内存带宽
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # 并行数据加载
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main
```

### 1.2 内存访问模式优化

```python
def coalesced_memory_access():
    """优化内存访问模式"""
    # 1. 使用向量化加载
    for k in T.vectorized(TILE_K):
        A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
        B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
    
    # 2. 合并内存访问
    T.copy(A[start:end], A_shared)  # 自动合并访问
    
    # 3. 避免银行冲突
    # 使用适当的步长避免共享内存银行冲突
    stride = 1  # 避免 32 的倍数
```

### 1.3 内存布局优化

```python
def layout_optimized_kernel():
    """使用布局提示优化内存访问"""
    from tilelang.intrinsics import make_mma_swizzle_layout
    
    # 布局注解
    T.annotate_layout({
        A_shared: make_mma_swizzle_layout(A_shared),
        B_shared: make_mma_swizzle_layout(B_shared),
    })
    
    # 启用光栅化
    T.use_swizzle(panel_size=10, enable=True)
```

## 2. 计算优化策略

### 2.1 软件流水线优化

```python
def pipelined_computation():
    """软件流水线优化"""
    # 1. 选择合适的流水线深度
    num_stages = 3  # 通常 2-4 个阶段效果最好
    
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
        # 2. 重叠内存操作和计算
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        
        # 3. 计算与下一次内存加载重叠
        T.gemm(A_shared, B_shared, C_local)
```

### 2.2 并行化策略

```python
def parallelization_optimization():
    """并行化优化"""
    # 1. 选择合适的并行粒度
    for i, j in T.Parallel(block_M, block_N):
        # 细粒度并行
    
    # 2. 向量化优化
    for k in T.vectorized(TILE_K):
        # 向量化操作
    
    # 3. 线程绑定优化
    tn = T.get_thread_binding(0)  # threadIdx.x
    tk = T.get_thread_binding(1)  # threadIdx.y
```

### 2.3 数据类型优化

```python
def dtype_optimization():
    """数据类型优化"""
    # 1. 使用混合精度
    input_dtype = "float16"    # 输入数据
    accum_dtype = "float"      # 累加器使用更高精度
    
    # 2. 类型转换优化
    result = A[i].astype(accum_dtype) * B[i].astype(accum_dtype)
    
    # 3. 避免不必要的类型转换
    # 在计算前统一转换，避免重复转换
```

## 3. 同步和线程安全

### 3.1 T.sync_threads() 使用规范

**⚠️ 严格禁止的使用场景：**

1. **条件分支中的同步**
```python
# ❌ 错误示例 - 会导致死锁
if condition:
    T.sync_threads()  # 只有部分线程执行，其他线程永远等待

# ✅ 正确做法 - 所有线程都执行同步
T.sync_threads()
if condition:
    # 同步后的操作
```

2. **循环中的条件同步**
```python
# ❌ 错误示例
for i in range(n):
    if tid < threshold:
        T.sync_threads()  # 死锁风险

# ✅ 正确做法
for i in range(n):
    T.sync_threads()  # 所有线程都同步
    if tid < threshold:
        # 同步后的操作
```

3. **共享内存分配后的条件同步**
```python
# ❌ 错误示例
if tid < N:
    shared_mem = T.alloc_shared((N,), dtype)
    T.sync_threads()  # 只有部分线程分配了共享内存

# ✅ 正确做法
shared_mem = T.alloc_shared((N,), dtype)
T.sync_threads()
if tid < N:
    # 使用共享内存
```

### 3.2 同步使用的最佳实践

1. **确保所有线程参与同步**
```python
# ✅ 正确的同步模式
T.sync_threads()  # 所有线程都必须执行
# 后续操作
```

2. **在共享内存操作前后同步**
```python
# ✅ 写入共享内存
shared_mem[tid] = value
T.sync_threads()  # 确保所有写入完成

# ✅ 读取共享内存
T.sync_threads()  # 确保所有写入完成
result = shared_mem[tid]
```

3. **避免不必要的同步**
```python
# ❌ 过度同步
for i in range(n):
    T.sync_threads()  # 不必要的同步

# ✅ 只在必要时同步
for i in range(n):
    # 计算操作
    if i == n-1:  # 只在最后一次迭代同步
        T.sync_threads()
```

### 3.3 避免同步问题的最佳实践

**✅ 推荐的内置函数（避免手动同步）：**

```python
# 1. 内置归约函数 - 推荐使用，避免手动同步
T.reduce_sum(input_tensor, output_tensor, dim=axis)     # 求和归约
T.reduce_max(input_tensor, output_tensor, dim=axis)     # 最大值归约
T.reduce_min(input_tensor, output_tensor, dim=axis)     # 最小值归约
T.reduce_mean(input_tensor, output_tensor, dim=axis)    # 平均值归约

# 2. 内置内存操作
T.copy(src, dst)  # 高效的内存复制
T.clear(tensor)   # 清零操作

# 3. 内置数学函数
T.rsqrt(x)        # 平方根倒数
T.sqrt(x)         # 平方根
T.exp(x)          # 指数函数
T.log(x)          # 对数函数
```

**⚠️ 尽量避免使用手动归约（可能会导致进程阻塞）：**

```python
# ❌ 绝对禁止：手动归约会导致线程卡死
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    T.sync_threads()  # 死锁风险，线程卡死
    stride //= 2

# ❌ 绝对禁止：条件分支中的同步
if condition:
    T.sync_threads()  # 死锁风险

# ❌ 绝对禁止：循环中的条件同步
for i in range(n):
    if tid < threshold:
        T.sync_threads()  # 死锁风险
```

**❌ 避免的手动同步模式：**

```python
# 错误：手动归约（需要同步）
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    T.sync_threads()  # 死锁风险
    stride //= 2

# 正确：使用内置归约
T.reduce_sum(input, output, dim=1)  # 无需同步
```

**✅ 推荐的并行计算模式：**

```python
# 1. 使用 T.Parallel 进行并行计算
for i, j in T.Parallel(M, N):
    result[i, j] = input[i, j] * scale[i]

# 2. 使用 T.vectorized 进行向量化
for i in T.vectorized(N):
    result[i] = input[i] * scale

# 3. 使用 T.Pipelined 进行流水线
for k in T.Pipelined(K, num_stages=3):
    T.copy(A[k], shared_A)
    T.gemm(shared_A, shared_B, result)
```

**内置归约函数使用示例：**

```python
# LayerNorm 中的归约应用
A_pow_local = T.alloc_fragment((M, N), "float32")
A_powsum = T.alloc_fragment((M,), "float32")

# ✅ 使用内置归约替代手动同步
T.reduce_sum(A_pow_local, A_powsum, dim=1)  # 按行求和

# Softmax 中的归约应用
A_exp = T.alloc_fragment((M, N), "float32")
A_sum = T.alloc_fragment((M,), "float32")

# ✅ 使用内置归约
T.reduce_sum(A_exp, A_sum, dim=1)  # 按行求和

# 最大值归一化
A_max = T.alloc_fragment((M,), "float32")
T.reduce_max(A_local, A_max, dim=1)  # 按行求最大值
```

**⚠️ 重要警告：**
- **绝对禁止使用手动归约**：会导致线程卡死，UTL打满但MEM低
- **必须使用内置归约函数**：`T.reduce_sum()`, `T.reduce_max()` 等
- **避免任何手动同步**：`T.sync_threads()` 在归约中极其危险

## 4. 最佳实践总结

### 4.1 通用优化原则

1. **先正确性后性能**：确保内核正确性后再优化性能
2. **内存优先**：优先优化内存访问模式
3. **同步安全**：严格遵循同步使用规范，避免死锁

### 4.2 性能优化检查清单

- [ ] 优化内存访问模式（合并访问、向量化）
- [ ] 合理使用软件流水线
- [ ] 选择合适的 tile 大小
- [ ] 使用混合精度计算
- [ ] **检查同步使用安全性**：确保所有线程都参与同步
- [ ] **避免条件分支中的同步**：防止死锁
- [ ] **正确使用线程索引**：使用 T.Parallel 而不是 T.get_thread_binding

### 4.3 常见性能陷阱

1. **过度分块**：过小的 tile 导致内存访问效率低
2. **流水线深度不当**：过深或过浅的流水线影响性能
3. **内存银行冲突**：共享内存访问模式不当
4. **类型转换开销**：频繁的类型转换影响性能
5. **同步开销**：不必要的线程同步
6. **同步死锁**：条件分支中的同步导致线程卡死
7. **线程索引错误**：使用错误的线程索引获取方式
8. **共享内存分配错误**：在条件分支中分配共享内存

