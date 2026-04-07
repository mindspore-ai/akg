---
name: triton-ascend-optimization
description: "Triton Ascend 性能优化通用策略汇总，包括块大小选择、Grid 1D化、核内循环、算子拆分、数值稳定性、特殊矩阵优化等。适用于需要提升已有内核性能的优化场景"
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

- [ ] **Grid 1D 化**: `grid=(CORE_NUM,)` + 核内交错循环 `for block_id in range(pid, total, CORE_NUM)`
- [ ] **核内循环**: 无需 for 的场景添加额外循环，编译器自动多级流水
- [ ] **尝试不同 BLOCK_SIZE**: 从较大 tile 开始，`ub overflow` 则缩小；配合 autotune 自动选优
- [ ] **算子拆分**: 复杂融合算子可拆为多 kernel 顺序执行，有时性能更优
- [ ] **Autotune**: 列多组 tile 参数配置（不含 num_warps/num_stages）
- [ ] **Reduction 用标量累加**: 每个核心标量累加 + 单次 atomic 写入
- [ ] **内存对齐**: matmul 的 K 维度按 512B 对齐提升带宽
- [ ] **避免 host 端 permute**: 非最后维 reduce 在 kernel 内用多维索引处理
- [ ] **隐式广播**: 用 `a[:, None] * b` 替代 `tl.broadcast_to`，减少临时 tensor
- [ ] **load 时直接 mask**: `tl.load(ptr, mask=m, other=0.0)` 优于先加载再 `tl.where`

## Reduction 优化

每个核心先局部标量累加，最后一次原子写入：

```python
core_sum = 0.0
for block_start in range(pid, total_blocks, CORE_NUM):
    data = tl.load(...)
    core_sum += tl.sum(data, axis=0)
tl.atomic_add(output_ptr, core_sum)
```

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
- matmul 使用 fp32 累加器：`acc = tl.zeros([M, N], dtype=tl.float32)`
- 最后再转回目标精度：`result = acc.to(tl.float16)`

