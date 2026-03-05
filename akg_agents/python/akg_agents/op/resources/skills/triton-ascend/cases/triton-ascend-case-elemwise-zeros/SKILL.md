---
name: triton-ascend-case-elemwise-zeros
description: "小shape张量创建（zeros/arange/full）优化：通过减少核数避免多核启动和调度开销，单核处理性能优于多核并行，适用于shape较小（千级元素）的elementwise创建张量场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# Zeros 创建张量优化案例

## 任务特征
- **操作类型**：Elemwise类型，包含torch的arange、full、zeros、zeros_like等创建张量的操作
- **数据尺寸**：(2, 256, 16)，数据shape较小
- **数据类型**：float32
- **任务特点**：可以按照轴的顺序（可flatten为一根轴），外层并行，内层向量化

## 优化：小shape少核处理

```python
# 内核代码
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
tl.store(output_ptr + offsets, zeros, mask=mask)
```

### 优化内容
- 通过设置BLOCK_SIZE的大小，来调整并行，提高性能
- Shape较小时，核数尽量减小，可以避免多核启动和调度开销

### 总结
1. 在Ascend平台上，shape较小的时候，核数尽量减小，可以避免多核启动和调度开销，实现性能优化
2. 对于单纯的Elementwise操作，将多根轴的元素展开为一根轴，然后在这根轴上进行切分
3. 将block分配给每个线程块，若UB存不下，可考虑多次切分
