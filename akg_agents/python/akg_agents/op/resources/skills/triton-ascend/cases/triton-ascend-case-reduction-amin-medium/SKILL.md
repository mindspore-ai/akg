---
name: triton-ascend-case-reduction-amin-medium
description: "大规模2D归约（amin）reduce轴很大优化：在优先占满UB前提下为reduce轴分配较大切分尺寸（BLOCK_SIZE_N=16384最优），减少循环次数但需权衡单次迭代负载，适用于非reduce轴中等、reduce轴很大（50万级元素）的场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 大规模 2D Amin 归约优化

## 任务特征
- **数据尺寸**：(2048, 262144)，非reduce轴中等，reduce轴很大

## 优化：reduce轴大切分

```python
# 错误：简单：循环内多次归约
row_min = float('inf')
for n_start in range(0, N, BLOCK_SIZE_N):
    curr_min = tl.min(data_block, 1)
    row_min = tl.minimum(curr_min, row_min)

# 正确：优化：维护矩阵结构
curr_min = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    curr_min = tl.minimum(data_block, curr_min)
row_min = tl.min(curr_min, 1)
```

## Autotune 配置

```python
# 1. reduce轴切分较大, UB用满 -> 2864.90 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048})

# 2-4. reduce轴切分逐渐增大，M切分相应减小 -> 性能逐渐提升
triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096})   # 2840.48 us
triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 8192})   # 2801.20 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16384})  # 2779.78 us 最优
```

### 总结
在优先占满UB前提下，为reduce轴分配较大切分尺寸，减少循环次数，但需权衡单次迭代计算负载。
