---
name: triton-ascend-case-reduction-weighted-swiglu
description: "3D融合算子（Weighted SwiGLU Backward）优化：Reshape降维将前两维合并简化并行策略，行二次切分避免超UB，在优先占满UB前提下为reduce轴分配较大切分尺寸，grid数较大时可能性能更优，适用于3D张量逐元素+reduce融合的场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# Weighted SwiGLU Backward 融合算子优化

## 任务特征
- **数据尺寸**：(16, 1024, 2048) × 3，3D融合算子
- **特点**：先逐元素操作，再reduce最后一根轴

## 优化 1：Reshape 降维

```python
# 将前两个维度(B, M)合并为单个维度BM
x_reshaped = x.reshape(BM, N)
weight_reshaped = weight.reshape(BM, N)
grad_reshaped = grad.reshape(BM, N)

weighted_x_reshaped = weighted_x.reshape(BM, N)
grad_weight_reshaped = grad_weight.reshape(BM, N)
grad_x_reshaped = grad_x.reshape(BM)
```

**优势**：简化并行策略，优化内存访问模式，提高内核执行效率。

## 优化 2：行二次切分

```python
for bm_start in range(0, BLOCK_SIZE_BM, SUB_BLOCK_SIZE_BM):
    bm_offsets = pid * BLOCK_SIZE_BM + bm_start + tl.arange(0, SUB_BLOCK_SIZE_BM)
    bm_mask = bm_offsets < BM
```

## Autotune 配置

```python
# （AI core=40）
# 1. grid=512>40，reduce轴切分较小，UB占满 -> 1105.84 us
triton.Config({'BLOCK_SIZE_BM': 32, 'SUB_BLOCK_SIZE_BM': 32, 'BLOCK_SIZE_N': 128})

# 2. grid=1024>40，reduce轴切分增至256 -> 1110.47 us
triton.Config({'BLOCK_SIZE_BM': 16, 'SUB_BLOCK_SIZE_BM': 16, 'BLOCK_SIZE_N': 256})

# 3. grid=2048>40，reduce轴切分增至512 -> 1091.26 us 最优
triton.Config({'BLOCK_SIZE_BM': 8, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512})

# 4. grid=32<40，reduce轴切分较大，UB占满 -> 1098.53 us
triton.Config({'BLOCK_SIZE_BM': 512, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512})

# 5. grid=40，有尾块，reduce轴切分较大，UB占满 -> 1094.60 us
triton.Config({'BLOCK_SIZE_BM': 416, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512})
```

### 总结
1. Reshape降维可简化并行策略，优化内存访问
2. 在优先占满UB前提下为reduce轴分配较大切分尺寸
3. Grid数较大时，可能性能更优（配置3）
