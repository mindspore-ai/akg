---
name: triton-ascend-case-matmul-large-k
description: "矩阵乘法矩阵乘法 A[M, K] @ B[K, N] = C[M, N]中，大K维度矩阵乘法(K>>M,N)优化：针对M/N较小但K极大(如M=N=256,K=131072)的场景，Split-K切分K维度并行化、Workspace+Reduce替代全局同步，实现显著性能提升"
category: deprecated
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

# 大 K 维度矩阵乘法优化案例

## 任务特征
- **操作类型**：矩阵乘法 A[M, K] @ B[K, N] = C[M, N]
- **典型数据尺寸**：A[256, 131072] @ B[131072, 256] = C[256, 256]
- **特点**：K 远大于 M 和 N（K/M = 512 倍），输出块数远少于核心数，常规 matmul 核心利用率低

### 核心问题

```
M=256, N=256, K=131072, BLOCK_M=64, BLOCK_N=64:
  输出块数 = ceil(256/64) × ceil(256/64) = 4 × 4 = 16
  可用核数 = 32
  → 16 块 < 32 核, 一半核空闲!
  → 每个核的 K-loop = 131072/256 = 512 次, 单核计算量极大
```


## 优化 1：Split-K + Atomic Add 并行化

### 原理

当输出块数 < 核心数时，将 K 维度切分成 `SPLIT_K` 段，让多个核并行计算同一输出块的不同 K 区间，用 `tl.atomic_add` 将划分后的partial结果累加到 C。另外，如果把`SPLIT_K`参数放在 grid 中，调整核数，可以使得无核空转。

```python
# grid = (NUM_MN_BLOCKS, SPLIT_K)
# 例如：AI_Cude=32，M=N=256, BLOCK=128: NUM_MN_BLOCKS = 2*2 = 4
# grid = (4, 16) → 64 , 32核每核处理2块数据
@triton.jit
def matmul_splitk_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)       # 输出块 ID
    split_id = tl.program_id(1)  # K 分段 ID

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(k_block_start, k_block_end):
        a = tl.load(A_ptr + ...)
        b = tl.load(B_ptr + ...)
        acc += tl.dot(a, b)

    # 原子加: 多个 split 的 partial 直接累加到 C
    tl.atomic_add(C_ptr + ..., acc, mask=...)
```

### 核心要点
- grid数的配置应接近或超过核心数，确保核满载
- `SPLIT_K` 越大并行度越高，但 atomic_add 竞争也越多

## 优化 2：Workspace + Reduce

### 原理

全局同步（如 `tl.debug_barrier`）会让所有核在同一点等待，等同于将 CUBE 计算和 VEC 归约完全串行化，性能极差。这里不像 AscendC 有 AIC/AIV 硬件并行操作实现，核内直接将 CUBE 结果写到 workspace，然后外部调用Reduce进行归约。另外，workspace的大小应该尽可能的装满，不要申请的过大。

```python

@triton.jit
def matmul_splitk_to_ws_kernel(A_ptr, B_ptr, WS_ptr, M, N, K, ...,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    # ... K 分段计算 ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(k_block_start, k_block_end):
        acc += tl.dot(tl.load(A_ptr + ...), tl.load(B_ptr + ...))

    # 直接 store 到 workspace, 不做任何归约
    tl.store(WS_ptr + split_id * stride_ws_s + ..., acc, mask=...)

# host 端
...
# 归约
C = torch.sum(workspace, dim=0)
```

### 核心要点
- Triton 中 CUBE (矩阵计算) 和 VEC (归约) 无法像 AscendC 那样通过 AIC/AIV 硬件通路真正并行
- `tl.debug_barrier` 全局同步将所有核阻塞，相当于串行化，性能最差
- 将归约提到 kernel 外部用 `torch.sum` 实现，避免了核内 CUBE-VEC 串行问题，实测比全局同步方案快 1 倍以上

## 总结

针对 K 远大于 M/N 的矩阵乘法场景（如 M=N=256, K=131072），三个优化可组合使用：

2. **Split-K + Atomic Add**：将 K 维度切分到 grid 外层维度，多核并行处理同一输出块的不同 K 段，用 `tl.atomic_add` 累加。
3. **Workspace + Reduce**：Split-K 各段写入 workspace 后，用 `torch.sum` 外部归约，避免核内全局同步的串行化问题。比 `debug_barrier` 方案快 1 倍以上
