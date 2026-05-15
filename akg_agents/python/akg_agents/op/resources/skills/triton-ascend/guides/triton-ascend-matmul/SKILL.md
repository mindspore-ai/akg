---
name: triton-ascend-matmul
description: "适用于矩阵乘法(matmul)类算子的优化指南。当算子的核心计算涉及二维或更高维的矩阵乘法时应选择此指南，典型算子包括：matmul, mm, bmm, linear, gemm, outer_product, einsum(含矩阵乘), conv(转为矩阵乘实现)等。涵盖 Cube Core 使用、分块(tiling)策略、Swizzle 优化、大 K 维处理等关键技巧。不适用于纯逐元素运算或纯归约运算。对于 attention 机制中的 QK^T 和 score*V 矩阵乘，若算子整体是注意力计算，应优先选择 attention 指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# MatMul 算子优化

> 适用于矩阵乘法及相关运算

## 核心数选择（硬约束）

- 涉及 `tl.dot` / 矩阵乘法运算 → **必须使用 CUBE_CORE_NUM**
- 混合运算（先 matmul 再 elementwise 后处理）→ **CUBE_CORE_NUM**
- 纯 elementwise / 标量运算 → VEC_CORE_NUM

**使用 VEC_CORE_NUM 启动 matmul kernel 会导致数值结果错误。**

## Tile Size 限制（硬件约束）

MatMul 数据走 L0A/L0B/L0C，tile 大小受硬件存储容量限制，超出会导致 `ub overflow` / `cbuf overflow` 编译错误。

约束公式（具体容量参考目标硬件信息文档）：
- L0A：`BLOCK_M × BLOCK_K × sizeof(dtype) ≤ L0A容量`
- L0B：`BLOCK_K × BLOCK_N × sizeof(dtype) ≤ L0B容量`
- L0C：`BLOCK_M × BLOCK_N × sizeof(acc_dtype) ≤ L0C容量`

遇 `ub overflow` / `cbuf overflow` → **缩小 BLOCK_M, BLOCK_N 或 BLOCK_K**

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
        try:
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.CUBE_CORE_NUM = 20

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        num_cores = self.CUBE_CORE_NUM
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

        matmul_kernel[(num_cores,)](
            a, b, c, M, N, K, num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        return c
```

**关键点**:
- 使用 `grid=(num_cores,)` 固定启动核心数（CUBE_CORE_NUM）
- 每个核心通过 `for block_idx in range(pid, NUM_BLOCKS, num_cores)` 循环处理多个块
- 不要使用 `grid=(NUM_BLOCKS_M * NUM_BLOCKS_N,)` 为每个块启动一个程序