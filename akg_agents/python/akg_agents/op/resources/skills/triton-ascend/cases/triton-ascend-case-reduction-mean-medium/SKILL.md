---
name: triton-ascend-case-reduction-mean-medium
description: "中等规模reduce第一根轴（mean）优化：计算重组减少归约次数，网格规模略小于AI Core数量且避免尾块时性能最佳（grid=32最优9.98us），适用于reduce第一根轴、两轴均中等（百万级元素）的2D归约场景"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 中等规模 Mean 归约优化（reduce第一根轴）

## 任务特征
- **数据尺寸**：(1024, 4096)，reduce第一根轴，非reduce轴中等

## 优化：计算重组

```python
# 简单
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
  错误：row_sum += tl.sum(block_vals)

# 正确：优化
col_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    col_sum += block_vals
col_sum = tl.sum(col_sum, axis=0)
```

## Autotune 配置

```python
# （AI core=40）
# 1. grid=16<40, UB占满 -> 13.32 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2. grid=40，有尾块 -> 35.12 us
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 103})

# 3. grid=32<40，UB占满 -> 9.98 us 最优
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128})

# 4. grid=64>40，UB占满 -> 13.33 us
triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64})

# 5. grid=128>40，UB占满 -> 22.22 us
triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 32})
```

### 总结
网格规模略小于AI Core数量且避免尾块时性能最佳。尾块导致性能大幅下降。
