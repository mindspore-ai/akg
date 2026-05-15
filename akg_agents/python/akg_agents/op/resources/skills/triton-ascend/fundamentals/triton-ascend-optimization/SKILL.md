---
name: triton-ascend-optimization
description: "Triton Ascend 性能优化通用策略: BLOCK_SIZE 选择 (1024-2048 for elementwise, must be <65536), grid configuration (use VEC_CORE_NUM / CUBE_CORE_NUM, 2D/3D grid for matmul / conv / reduce, 1D grid + inner loop for elementwise / pointwise), 256B alignment for memory transfers, autotune block-size patterns, fp16 / fp32 precision conversion. Bind via keywords like matmul, elementwise, reduce, block_size, grid, autotune, alignment, fp16, fp32, tile, interleaved-loop, cube-core, vec-core."
category: guide
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
- [ ] **Grid 维度选择**:
  - 对于计算密集型算子（矩阵乘、卷积、大块 reduce等），考虑 2D/3D grid，利用硬件调度优势
  - 对于大量小规模计算（element-wise、pointwise等），考虑 1D grid + 核内循环，减少启动开销
- [ ] **核内循环**: 无需 for 的场景添加额外循环，编译器自动多级流水
- [ ] **尝试不同 BLOCK_SIZE**: 从较大 tile 开始，`ub overflow` 则缩小；在核内循环中平衡并行度和资源占用：
  - 尝试小切分策略，使读写能够并行进行
  - 尝试大切分策略，提升 UB 使用率
  - 列多组参数配置，添加 @triton.autotune
- [ ] **算子拆分**: 复杂融合算子可拆为多 kernel 顺序执行，有时性能更优
- [ ] **Autotune**: 列多组 tile 参数配置（不含 num_warps/num_stages）
- [ ] **Reduction 用标量累加**: 每个核心标量累加 + 单次 atomic 写入
- [ ] **内存对齐**: matmul 的 K 维度按 512B 对齐提升带宽
- [ ] **避免 host 端 permute**: 非最后维 reduce 在 kernel 内用多维索引处理
- [ ] **隐式广播**: 用 `a[:, None] * b` 替代 `tl.broadcast_to`，减少临时 tensor
- [ ] **load 时直接 mask**: `tl.load(ptr, mask=m, other=0.0)` 优于先加载再 `tl.where`
- [ ] **减少冗余精度转换**: 避免反复在 fp16/fp32 转换 即`.to(float16)`和`.to(float32)`，一次转换多次复用
- [ ] **核心数配置**: grid 数设为核心数（VEC/CUBE），过大时启动开销反增
- [ ] **256B 对齐**: 数据搬运以 256B 为单位，对齐可提升带宽

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
