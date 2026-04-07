---
name: triton-ascend-optimization
description: "Triton Ascend 性能优化通用策略汇总，包括块大小选择、Grid 1D化、核内循环、算子拆分、数值稳定性等优化 checklist。适用于需要提升已有内核性能的优化场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
structure:
  child_skills:
    - triton-ascend-memory
    - triton-ascend-grid-config
    - triton-ascend-debugging
---

# Triton Ascend 性能优化指南

## 优化策略 Checklist

- [ ] **Grid 1D 化**: 将 2D grid 改为 1D + 核内循环，降低启动开销
- [ ] **核内循环**: 无需 for 的场景添加额外循环，编译器自动多级流水
- [ ] **尝试不同 BLOCK_SIZE**: 在核内循环中平衡并行度和资源占用
- [ ] **算子拆分**: 复杂融合算子可拆为多 kernel 顺序执行，有时性能更优
- [ ] **Autotune**: 列多组参数配置，添加 @triton.autotune

## 数值稳定性

### 防溢出
```python
max_val = tl.max(scores, axis=0)
scores = scores - max_val
p = tl.math.exp2(scores)
```

### 防负值开方
- 任何 sqrt 前确保非负: `max(input, 0.)` 或 `max(input, eps)`

### 精度提升
- 使用 float32 累加，最后再转回目标精度

