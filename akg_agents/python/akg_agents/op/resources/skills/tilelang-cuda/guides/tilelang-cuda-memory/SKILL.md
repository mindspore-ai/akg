---
name: tilelang-cuda-memory
description: "TileLang CUDA 内存访问优化策略，包括 T.alloc_shared/fragment 使用、数据布局优化、合并访存和 Bank Conflict 避免技巧。适用于内存带宽受限、需要优化数据搬运效率的 TileLang 内核性能优化场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
---

# TileLang CUDA 内存访问优化

内存访问是 GPU 性能的关键瓶颈。本文档提供 TileLang CUDA 的内存访问优化策略。

---

## 1. GPU 内存层次

### 内存带宽和延迟

| 内存类型 | 带宽 (A100) | 延迟 | 容量 | TileLang API |
|---------|-------------|------|------|-------------|
| 寄存器 | ~19 TB/s | 1 cycle | 256 KB/SM | `T.alloc_fragment` / `T.alloc_local` |
| 共享内存 | ~19 TB/s | ~20 cycles | 164 KB/SM | `T.alloc_shared` |
| L2 缓存 | ~5 TB/s | ~100 cycles | 40 MB | `T.use_swizzle` 优化 |
| 全局内存 (HBM) | ~2 TB/s | ~400 cycles | 40/80 GB | `T.Tensor` |

### 优化原则
- **减少全局内存访问**: 利用共享内存和寄存器缓存数据
- **合并访问 (Coalesced Access)**: 使用 `T.copy` 进行高效数据传输
- **提高 L2 缓存命中率**: 使用 `T.use_swizzle` 优化数据局部性

---

## 2. 内存分配最佳实践

### 共享内存（频繁访问的数据）

```python
# 共享内存用于缓存从全局内存加载的数据块
A_shared = T.alloc_shared((block_M, block_K), "float16")
B_shared = T.alloc_shared((block_K, block_N), "float16")

# 高效数据加载
T.copy(A[by * block_M, ko * block_K], A_shared)
```

**适用场景**：
- 矩阵乘法中的输入块
- 多次访问的中间数据
- 需要线程间共享的数据

### 寄存器片段（累加器和临时存储）

```python
# 寄存器用于累加和局部计算
C_local = T.alloc_fragment((block_M, block_N), "float")
T.clear(C_local)

# 累加操作
T.gemm(A_shared, B_shared, C_local)
```

**适用场景**：
- 矩阵乘法累加器
- 归约的临时结果
- 局部计算结果

### 本地内存（线程私有存储）

```python
# 本地内存用于线程私有变量
C_reg = T.alloc_local((1,), "float")
T.clear(C_reg)
```

**适用场景**：
- 单个线程的累加值
- 线程局部的临时变量

---

## 3. 数据传输优化

### 使用 T.copy 进行合并访问

```python
# ✅ 推荐：使用 T.copy 自动合并访问
T.copy(A[by * block_M, ko * block_K], A_shared)
T.copy(B[ko * block_K, bx * block_N], B_shared)

# ✅ 结果写回
T.copy(C_local, C[by * block_M, bx * block_N])
```

### 使用 T.Parallel 进行并行数据复制

```python
# 并行数据复制
for k, j in T.Parallel(block_K, block_N):
    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
```

### 使用向量化加载

```python
# 向量化加载以提高带宽利用率
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
```

---

## 4. 软件流水线

### 基本用法

```python
# 使用 T.Pipelined 实现软件流水线
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    # 数据加载和计算自动重叠
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

### num_stages 选择指南

| num_stages | 共享内存使用 | 性能 | 适用场景 |
|-----------|-------------|------|---------|
| 2 | 最少 | 基础 | 共享内存紧张时 |
| 3 | 中等 | 通常最优 | 默认推荐 |
| 4 | 较多 | 大矩阵时更优 | 共享内存充足时 |
| 5+ | 很多 | 可能下降 | 需要测试验证 |

---

## 5. L2 缓存优化

### Swizzle 光栅化

通过 `T.use_swizzle` 改善 L2 缓存局部性，特别适用于矩阵乘法等 2D 计算：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # 启用 swizzle 以提高 L2 缓存命中率
    T.use_swizzle(panel_size=10, enable=True)
    
    # ... 计算逻辑 ...
```

### 布局注解

```python
from tilelang.intrinsics import make_mma_swizzle_layout

# 使用布局注解优化共享内存访问模式
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
```

---

## 6. 分块大小选择

### 推荐设置

| 算子类型 | block 大小 | 线程数 | 说明 |
|---------|-----------|-------|------|
| Element-wise | block=256~1024 | 128-256 | 一维并行 |
| GEMM | M=128, N=128, K=32 | 128 | 二维分块 |
| GEMV | N=64~256, K=32~128 | 128 | 一维 + 串行 |
| Reduce | block=256~512 | 128-256 | 归约维度分块 |
| LayerNorm | block=256 | 256 | 按行处理 |

### 选择原则
- **平衡并行度与资源占用**: 避免过大或过小
- **使用 2 的幂次**: 便于硬件优化
- **考虑共享内存限制**: 每个 SM 通常 164 KB
- **考虑寄存器压力**: 过大的 fragment 可能导致溢出

---

## 7. 完整优化示例

### 优化的矩阵乘法

```python
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

@tilelang.jit(out_idx=[-1])
def optimized_matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 1. 内存分配
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            # 2. 布局优化
            T.annotate_layout({
                A_shared: make_mma_swizzle_layout(A_shared),
                B_shared: make_mma_swizzle_layout(B_shared),
            })
            
            # 3. L2 缓存优化
            T.use_swizzle(panel_size=10, enable=True)
            
            T.clear(C_local)
            
            # 4. 软件流水线
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # 5. 结果写回
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main
```

---

## 8. 最佳实践总结

### 内存分配
1. **共享内存**: 用于频繁访问的数据块
2. **寄存器片段**: 用于累加器和临时计算
3. **本地内存**: 用于线程私有变量

### 数据传输
1. **使用 T.copy**: 自动合并内存访问
2. **使用 T.Pipelined**: 重叠数据加载和计算
3. **使用 T.vectorized**: 向量化数据加载

### 缓存优化
1. **T.use_swizzle**: 优化 L2 缓存局部性
2. **T.annotate_layout**: 优化共享内存访问模式
3. **合理分块**: 平衡并行度和缓存利用

### 避免的陷阱
- 过大的共享内存分配导致 occupancy 下降
- 忽略软件流水线优化
- 分块大小设置不当导致内存访问效率低
- 忘记 L2 缓存优化
