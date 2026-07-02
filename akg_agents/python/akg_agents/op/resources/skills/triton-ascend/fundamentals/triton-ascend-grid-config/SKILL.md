---
name: triton-ascend-grid-config
description: "Grid/Block 配置策略，包括核数选择、并行度调优、二次切分和大 shape 算子处理方案。适用于需要确定 kernel 启动参数、优化多核并行效率、或处理超大规模数据的内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Grid 配置策略

## Grid 限制
- Grid 必须是 tuple，最多 3 维：`(x,)`, `(x, y)`, `(x, y, z)`
- 当前 910B 运行环境默认开启 `TRITON_ALL_BLOCKS_PARALLEL=1`，允许超过 65535 的一维逻辑 grid 由后端映射到物理核心循环
- 若逻辑 tile 是 2D/3D，请优先展平成 1D grid，并在 kernel 内反解坐标；不要直接依赖超大 2D/3D grid
- BLOCK_SIZE 仍需满足 UB/L0 和编译器限制；不要为了降低 grid 数而放大每个 program 的 tile

## 推荐方案 1：一维 flattened 逻辑 Grid

适用于原本需要 `(num_m_tiles, num_n_tiles)` 这类多维 tile grid 的算子。保留小 tile，使用一维 grid 启动：

```python
num_m_tiles = triton.cdiv(M, BLOCK_M)
num_n_tiles = triton.cdiv(N, BLOCK_N)
grid = (num_m_tiles * num_n_tiles,)

@triton.jit
def kernel(..., NUM_N_TILES: tl.constexpr, ...):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N_TILES
    pid_n = pid - pid_m * NUM_N_TILES
    # 处理(pid_m, pid_n)对应的小tile
```

## 推荐方案 2：交错循环（固定 Grid 为核心数）

适用于按行/按块独立处理的算子（Element-wise、Reduce、Normalization 等）。

```python
@triton.jit
def kernel(
    input_ptr, output_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    # 交错处理：pid=0 处理第 0, CORE_NUM, 2*CORE_NUM, ... 行
    for row_idx in range(pid, M, CORE_NUM):
        row_ptr = input_ptr + row_idx * stride_m
        out_ptr = output_ptr + row_idx * stride_m
        for col_start in range(0, N, BLOCK_N):
            offs = col_start + tl.arange(0, BLOCK_N)
            mask = offs < N
            data = tl.load(row_ptr + offs * stride_n, mask=mask)
            result = compute(data)
            tl.store(out_ptr + offs * stride_n, result, mask=mask)
```

## 动态获取核心数

**必须在 `__init__` 中获取**，禁止在 forward 中调用（触发设备同步）。

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.VEC_CORE_NUM = 40
            self.CUBE_CORE_NUM = 20

    def forward(self, x):
        M, N = x.shape
        out = torch.empty_like(x)
        grid = (self.VEC_CORE_NUM,)
        kernel[grid](x, out, M, N, x.stride(0), x.stride(1),
                     BLOCK_N=256, CORE_NUM=self.VEC_CORE_NUM)
        return out
```

### 核心数选择
- **向量算子**（element-wise、softmax、归一化）：使用 `VEC_CORE_NUM`
- **矩阵算子**（matmul、attention）：使用 `CUBE_CORE_NUM`

## 多次切分策略

若出现 `ub overflow` 或单次切分超硬件缓存，优先减小每个 program 的 tile，或嵌套循环做多层切分：

```python
for m_start in range(pid_m * BLOCK_M, min((pid_m + 1) * BLOCK_M, M), SUB_BLOCK_M):
    for n_start in range(pid_n * BLOCK_N, min((pid_n + 1) * BLOCK_N, N), SUB_BLOCK_N):
        # 处理 SUB_BLOCK 大小的子块
```
