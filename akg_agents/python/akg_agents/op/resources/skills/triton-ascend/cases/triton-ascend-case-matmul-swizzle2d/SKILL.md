---
name: triton-ascend-case-matmul-swizzle2d
description: "大矩阵乘法Swizzle2D优化：固定核心数启动（grid=20而非所有块）+Swizzle2D块重排（GROUP_SIZE=4）提升缓存局部性，根据M/N比例自适应选择分组方向，适用于大规模矩阵乘法（千万级元素）的Ascend NPU场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 矩阵乘法 Swizzle2D 优化案例

## 任务特征
- **操作类型**：矩阵乘法 A[M, K] @ B[K, N] = C[M, N]
- **数据尺寸**：A[2048, 7168] @ B[7168, 16384] = C[2048, 16384]
- **特点**：计算密集型，核心分配策略对缓存命中率和负载均衡影响显著

## 优化 1：固定核心数启动（最重要！）

### 错误：错误：启动所有块
```python
grid = (NUM_BLOCKS_M * NUM_BLOCKS_N,)  # 启动1024个程序
```

### 正确：正确：固定核心数启动
```python
num_cores = 20  # Ascend 910B4有20个AI Core

@triton.jit
def matmul_kernel(..., num_cores: tl.constexpr):
    pid = tl.program_id(axis=0)  # 0~19
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 处理块...
        pass

matmul_kernel[(num_cores,)](...)  # grid=(20,)
```

**核心要点**：Ascend NPU必须使用固定核心数启动，每个核心循环处理多个块。

## 优化 2：Swizzle2D 块重排

```python
@triton.jit
def matmul_kernel_swizzle2d(..., GROUP_SIZE: tl.constexpr, DIRECTION: tl.constexpr):
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        
        if DIRECTION == 0:  # M≥N: 行优先分组
            task_m_idx, task_n_idx = tl.swizzle2d(
                block_m, block_n, NUM_BLOCKS_M, NUM_BLOCKS_N, GROUP_SIZE
            )
        else:  # M<N: 列优先分组（手动实现）
            size_gj = GROUP_SIZE * NUM_BLOCKS_M
            group_id = block_idx // size_gj
            off_n = group_id * GROUP_SIZE
            cur_size_g = tl.minimum(NUM_BLOCKS_N - off_n, GROUP_SIZE)
            local_ij = block_idx % size_gj
            task_m_idx = local_ij // cur_size_g
            task_n_idx = off_n + local_ij % cur_size_g
```

### 优化内容
- Swizzle2D通过GROUP_SIZE将块按组重排，组内块共享数据
- GROUP_SIZE推荐值为4，可通过autotune搜索[1,2,3,4,5,8]

## 优化 3：矩阵形状自适应

```python
DIRECTION = 1 if m < n else 0  # M<N列优先, M≥N行优先
```

- **M≥N时**：行优先分组，减少mat_a重复加载
- **M<N时**：列优先分组，减少mat_b重复加载

## 优化 4：分块大小选择

```python
# float16/bfloat16
BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 256

# float32
BLOCK_M, BLOCK_K, BLOCK_N = 128, 128, 128
```

### 总结
1. **固定核心数启动**：`grid=(20,)`，每个核心循环处理多个块
2. **Swizzle2D 重排**：通过块分组提升缓存局部性
3. **自适应分组方向**：根据M/N比例选择行优先或列优先
4. **合适的分块大小**：根据数据类型和缓存容量选择
