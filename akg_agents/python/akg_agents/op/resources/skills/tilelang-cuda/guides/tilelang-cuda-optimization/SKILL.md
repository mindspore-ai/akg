---
name: tilelang-cuda-optimization
description: "TileLang CUDA 性能优化通用策略、最佳实践和调试技巧汇总。适用于需要提升 TileLang 内核性能、遇到编译/运行错误需要排查、或需要了解 TileLang 平台限制的内核代码生成和优化场景"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
structure:
  child_skills:
    - tilelang-cuda-memory
    - tilelang-cuda-synchronization
---

# TileLang CUDA 性能优化指南

## 1. 性能优化策略

### 1.1 分块大小选择

- **原则**: 平衡并行度与资源占用
- **建议**: 使用 2 的幂次
- **常用值**: block_M/block_N = 64, 128, 256; block_K = 16, 32, 64

| 算子类型 | 推荐分块大小 | 线程数 |
|---------|-------------|-------|
| Element-wise | block = 256-1024 | 128-256 |
| GEMM | block_M=128, block_N=128, block_K=32 | 128 |
| Reduce | block = 256-512 | 128-256 |

### 1.2 软件流水线优化

```python
def pipelined_computation():
    # 选择合适的流水线深度
    num_stages = 3  # 通常 2-4 个阶段效果最好
    
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
        # 重叠内存操作和计算
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
```

**流水线深度选择**：
- `num_stages=2`: 最少的共享内存使用
- `num_stages=3`: 通常最优（推荐默认值）
- `num_stages=4`: 更多重叠但占用更多共享内存
- `num_stages=5+`: 可能超出共享内存限制

### 1.3 并行化策略

```python
# 1. 细粒度并行
for i, j in T.Parallel(block_M, block_N):
    # 自动映射到线程
    pass

# 2. 向量化优化
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]

# 3. 串行循环（必要时使用）
for k in T.serial(block_K):
    # 顺序执行
    pass
```

### 1.4 数据类型优化

```python
# 1. 使用混合精度
input_dtype = "float16"    # 输入数据
accum_dtype = "float"      # 累加器使用更高精度

# 2. 类型转换优化
result = A[i].astype(accum_dtype) * B[i].astype(accum_dtype)

# 3. 避免不必要的类型转换（在计算前统一转换）
```

## 2. 内存优化策略

### 2.1 内存层次结构优化

```python
def memory_optimized_matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 1. 共享内存分配 - 缓存频繁访问的数据
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            
            # 2. 寄存器片段分配 - 累加和临时存储
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            # 3. 启用 swizzle 以提高 L2 缓存局部性
            T.use_swizzle(panel_size=10, enable=True)
            
            T.clear(C_local)
            
            # 4. 软件流水线优化内存带宽
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main
```

### 2.2 内存访问模式优化

```python
# 1. 使用向量化加载
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]

# 2. 合并内存访问
T.copy(A[start:end], A_shared)  # 自动合并访问

# 3. 使用布局优化避免银行冲突
from tilelang.intrinsics import make_mma_swizzle_layout
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
```

### 2.3 L2 缓存优化

```python
# 启用光栅化以提高 L2 缓存局部性
T.use_swizzle(panel_size=10, enable=True)
```

## 3. 同步安全优化

### 3.1 使用内置归约函数（强烈推荐）

```python
# ✅ 推荐：使用内置归约函数，无需手动同步
T.reduce_sum(input_tensor, output_tensor, dim=axis)
T.reduce_max(input_tensor, output_tensor, dim=axis)
T.reduce_min(input_tensor, output_tensor, dim=axis)
T.reduce_mean(input_tensor, output_tensor, dim=axis)
```

### 3.2 避免手动归约

```python
# ❌ 绝对禁止：手动归约会导致线程卡死
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    T.sync_threads()  # 死锁风险
    stride //= 2

# ✅ 正确：使用内置归约
T.reduce_sum(input, output, dim=1)  # 无需同步
```

### 3.3 推荐的并行计算模式

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

## 4. 数值稳定性

### 4.1 防溢出处理

```python
# Softmax 数值稳定化
T.fill(scores_max, -T.infinity("float"))
T.reduce_max(acc_s, scores_max, dim=1)
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

### 4.2 精度提升

- **使用 float32 进行累加**: 即使输入是 float16/bfloat16
- **最后再转换**: 计算完成后再转回目标精度

```python
# 累加器使用高精度
C_local = T.alloc_fragment((block_M, block_N), "float")  # float32 累加
# 计算完成后转换
result = C_local.astype("float16")
```

## 5. 性能检查清单

### 内存访问
- [ ] 优化内存访问模式（合并访问、向量化）
- [ ] 合理使用共享内存缓存数据
- [ ] 启用 swizzle 优化 L2 缓存

### 并行度配置
- [ ] 分块大小是否合理（2 的幂次）
- [ ] 线程数是否合适（128/256）
- [ ] 软件流水线深度是否合适（2-4）

### 计算优化
- [ ] 合理使用软件流水线 `T.Pipelined`
- [ ] 使用混合精度计算
- [ ] 利用内置原语（T.gemm, T.reduce_* 等）

### 安全性
- [ ] **检查同步使用安全性**: 确保所有线程都参与同步
- [ ] **避免条件分支中的同步**: 防止死锁
- [ ] **使用内置归约函数**: 避免手动归约导致的线程卡死
- [ ] **正确使用线程索引**: 使用 T.Parallel 而不是 T.get_thread_binding

### 数值稳定性
- [ ] Reduce 操作是否有防溢出处理
- [ ] 是否使用 float32 进行中间累加
- [ ] 是否处理了除零、负数开方等边界情况

## 6. 常见性能陷阱

1. **过度分块**: 过小的 tile 导致内存访问效率低
2. **流水线深度不当**: 过深或过浅的流水线影响性能
3. **内存银行冲突**: 共享内存访问模式不当
4. **类型转换开销**: 频繁的类型转换影响性能
5. **同步开销**: 不必要的线程同步
6. **同步死锁**: 条件分支中的同步导致线程卡死
7. **线程索引错误**: 使用错误的线程索引获取方式
8. **共享内存分配错误**: 在条件分支中分配共享内存

## 最佳实践总结

1. **先正确性后性能**: 确保内核正确性后再优化性能
2. **内存优先**: 优先优化内存访问模式
3. **同步安全**: 严格遵循同步使用规范，避免死锁
4. **使用内置原语**: 优先使用 T.gemm、T.reduce_* 等内置函数
5. **混合精度**: 输入用低精度，累加用高精度
6. **流水线**: 通过 T.Pipelined 隐藏内存延迟
