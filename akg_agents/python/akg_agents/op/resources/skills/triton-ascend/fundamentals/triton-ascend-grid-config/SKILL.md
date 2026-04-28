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
- 各维度乘积不超过 65535
- BLOCK_SIZE 必须小于 65536

## 推荐方案：交错循环（固定 Grid 为核心数）

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

若 BLOCK_SIZE 超限或单次切分超硬件缓存，可嵌套循环做多层切分：

```python
for m_start in range(pid_m * BLOCK_M, min((pid_m + 1) * BLOCK_M, M), SUB_BLOCK_M):
    for n_start in range(pid_n * BLOCK_N, min((pid_n + 1) * BLOCK_N, N), SUB_BLOCK_N):
        # 处理 SUB_BLOCK 大小的子块
```
