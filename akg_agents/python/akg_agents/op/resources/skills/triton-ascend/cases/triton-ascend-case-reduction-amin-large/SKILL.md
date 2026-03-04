---
name: triton-ascend-case-reduction-amin-large
description: "极大规模1D归约（amin）优化：二次切分避免超UB+计算重组减少归约次数，网格数接近AI Core数量（grid=32）、UB用满、无尾块时性能最优（9.61us），适用于极大规模1D数据（400万级元素）的全量归约场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 极大规模 1D Amin 归约优化

## 任务特征
- **数据尺寸**：(4194304,)，极大规模1D数据

## 优化 1：二次切分

```python
pid = tl.program_id(0)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = pid * BLOCK_SIZE + start + tl.arange(0, SUB_BLOCK_SIZE)
```

## 优化 2：计算重组

```python
# 错误：简单
row_min = float('inf')
for n_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    curr_min = tl.min(block_data)
    row_min = tl.minimum(curr_min, row_min)

# 正确：优化
curr_min = tl.full((SUB_BLOCK_SIZE,), float('inf'), dtype=tl.float32)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    curr_min = tl.minimum(curr_min, block_data)
min_val = tl.min(curr_min)
```

## Autotune 配置

```python
# （AI core=40）
# 1. grid=16<40, UB用满 -> 15.12 us
triton.Config({'BLOCK_SIZE': 262144, 'SUB_BLOCK_SIZE': 16384})

# 2. grid=32<40, UB用满 -> 9.61 us 最优
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 16384})

# 3. grid=32, UB未用满 -> 10.29 us
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 8192})

# 4. grid=40, UB用满, 有尾块 -> 10.17 us
triton.Config({'BLOCK_SIZE': 104858, 'SUB_BLOCK_SIZE': 16384})

# 5. grid=64>40, UB用满 -> 11.64 us
triton.Config({'BLOCK_SIZE': 65536, 'SUB_BLOCK_SIZE': 32768})
```

### 总结
网格数接近AI Core数量、UB用满、无尾块时性能最优。二次切分避免超出硬件缓存。
