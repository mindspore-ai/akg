---
name: triton-ascend-case-reduction-amax-small
description: "极小规模归约（amax）优化：单核处理（grid=1）优于多核并行（2.16us vs 3.51us），避免并行化带来的调度开销，适用于数据规模很小（<1000元素）的归约场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 极小规模 Amax 归约优化

## 任务特征
- **数据尺寸**：(16, 16)，非常小
- **策略**：单核处理优于多核并行

## 优化：单核/少核处理

```python
# 1. 单核处理 -> 性能：2.16 us 最优
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16})

# 2. 多核并行 -> 性能：3.51 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16})
```

### 总结
对于小规模计算任务，单核（少核）处理性能更优，并行化带来的开销超过了收益。
