---
name: triton-ascend-case-reduction-prod-small
description: "小规模reduce第一根轴（prod）优化：使用自定义mul函数配合tl.reduce实现连乘（triton无prod接口），最优网格数明显小于AI Core数量（grid=16最优2.15us），过高并行度反而因调度开销降低性能，适用于shape较小（10万级元素）的reduce第一根轴场景"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 小规模 Prod 归约优化

## 任务特征
- **数据尺寸**：(16, 2048)，reduce第一根轴，非reduce轴中等

## 优化：自定义reduce函数

```python
# 简单
accumulator = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
for m in range(M):
  错误：accumulator = tl.where(mask, accumulator * data, accumulator)

# 正确：优化
@triton.jit
def mul(a, b):
    return a * b

col_prod = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    col_prod *= block_vals
col_prod = tl.reduce(col_prod, axis=0, combine_fn=mul)  # triton没有prod接口
```

## Autotune 配置

```python
# （AI core=40）
# 1. grid=64>40 -> 4.21 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32})

# 2. grid=40，有尾块 -> 3.28 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 52})

# 3. grid=32<40 -> 2.61 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64})

# 4. grid=16<40 -> 2.15 us 最优
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128})

# 5. grid=2<40，UB占满 -> 2.63 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024})

# 6. grid=1 -> 3.25 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048})
```

### 总结
算子shape较小时（10^5~10^6元素），最优网格数可能需明显小于AI Core数量，过高并行度反而会因调度开销降低性能。
