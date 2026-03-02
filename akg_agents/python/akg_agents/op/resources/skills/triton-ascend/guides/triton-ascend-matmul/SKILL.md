---
name: triton-ascend-matmul
description: "矩阵乘法算子(matmul/bmm/linear)优化策略，包括分块切分、Swizzle 优化、Cube Core 利用和大矩阵处理技巧。适用于实现 GEMM、批量矩阵乘、全连接层等矩阵运算的内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  operator_patterns: "matmul"
  algorithms: "matmul, bmm, linear"
---

# MatMul 算子优化

> 适用于矩阵乘法及相关运算

## Ascend 后端切分优化

**关键原则**: 充分发挥带宽，算子行宽为 512B 的整数倍。

以 fp16/bf16 为例（每个元素 2 字节）：

### 切分配置（根据转置情况）

1. **A、B 都不转置**
   - 分块行宽分别为 K0 和 N0
   - **推荐**: M0=128, K0=256, N0=256

2. **A 不转置，B 转置**
   - 分块行宽都是 K0
   - **推荐**: K0=256, M0 和 N0 影响较小

3. **A、B 都转置**
   - 分块行宽分别为 M0 和 K0
   - **推荐**: M0=256, K0=256, N0=128

4. **A 转置，B 不转置**
   - 分块行宽分别为 M0 和 N0
   - **注意**: 左右矩阵均无法同时满足 512B 的整数倍，需根据实际情况调整

### 为什么是 512B？

- 512B = 256 个 fp16/bf16 元素（256 × 2 字节）
- NPU 的最佳带宽对齐
- 确保每次内存访问充分利用带宽

## 固定核心数启动

MatMul 算子使用 **CUBE核心数**（矩阵计算核心）。

**关键**: 使用 `grid=(num_cores,)` 而非 `(NUM_BLOCKS,)`

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

**关键点**:
- 使用 `grid=(num_cores,)` 固定启动核心数(如20个)
- 每个核心通过 `for block_idx in range(pid, NUM_BLOCKS, num_cores)` 循环处理多个块
- 不要使用 `grid=(NUM_BLOCKS_M * NUM_BLOCKS_N,)` 为每个块启动一个程序