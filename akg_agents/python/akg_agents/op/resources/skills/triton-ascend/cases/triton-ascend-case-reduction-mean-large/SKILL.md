---
name: triton-ascend-case-reduction-mean-large
description: "大规模reduce最后根轴（mean）行二次切分优化：每个kernel计算多行减少线程块数量、kernel内二次切分避免超UB，grid=40且SUB切分不含尾块时性能最优（16.00us），尾块计算会显著降低性能，适用于非reduce轴中等、reduce轴较大的2D归约场景"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 大规模 Mean 归约优化（行二次切分）

## 任务特征
- **数据尺寸**：(1000, 8192)，非reduce轴中等，reduce轴较大

## 优化：行二次切分

```python
pid = tl.program_id(0)
for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
    m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
```

**目的**：
- 每个kernel计算多行（BLOCK_SIZE_M），减少总线程块数量
- Kernel内对行进行二次切分（SUB_BLOCK_SIZE_M），避免超出硬件缓存

## Autotune 配置

```python
# （AI core=40）
# 1. grid<40 -> 28.64 us
triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512})

# 2. grid=40，SUB切分含尾块 -> 16.54 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096})

# 3. grid=40，SUB切分不含尾块 -> 16.00 us 最优
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512})

# 4. grid>40，且非核数整数倍 -> 25.86 us
triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 512})
```

### 总结
1. grid等于核数，SUB切分不含尾块时性能最优
2. 尾块计算会降低性能
3. grid超出核数且非核数整数倍时，各核计算任务不均匀，性能较差
