---
name: triton-ascend-optimization
description: "Triton Ascend 性能优化通用策略、API 限制说明和调试技巧汇总。适用于需要提升内核性能、遇到编译/运行错误需要排查、或需要了解 Ascend 平台特有限制的内核代码生成和优化场景"
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

- [ ] **块大小选择**: 1024-2048 for element-wise; BLOCK_SIZE < 65536
- [ ] **Grid 1D 化**: 将 2D grid 改为 1D + 核内循环，降低启动开销
- [ ] **核内循环**: 无需 for 的场景添加额外循环，编译器自动多级流水
- [ ] **尝试不同 BLOCK_SIZE**: 在核内循环中平衡并行度和资源占用
- [ ] **算子拆分**: 复杂融合算子可拆为多 kernel 顺序执行，有时性能更优
- [ ] **核心数配置**: grid 数设为核心数（VEC/CUBE），过大时启动开销反增
- [ ] **256B 对齐**: 数据搬运以 256B 为单位，对齐可提升带宽
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

## API 使用限制

### 禁止使用的语法
- `return` / `break` / `continue` → 使用 mask 控制
- lambda → 内联函数或 tl.where
- 链式布尔运算 → 分步计算 mask
- 张量直接索引 → tl.load / tl.store
- if-else 中负偏移 → tl.maximum(offset, 0)
- Ascend: 复杂 tl.where → if-else
- Ascend: while 循环 → for 替代
- Ascend: range() 的 start/stop 混用运行时变量和 constexpr → 用全 constexpr 的 range + 循环体内运行时 if 跳过

### While 循环替代（Ascend）

**静态上限**（编译时常量）: 直接 `for i in range(N_ITERS)`

**动态上限**（运行时参数）:
```python
@triton.jit
def kernel(ptr, n_iters, TILE: tl.constexpr, MAX_ITERS: tl.constexpr):
    for i in range(MAX_ITERS):
        if i < n_iters:
            offset = i * TILE + tl.arange(0, TILE)
            data = tl.load(ptr + offset)
            tl.store(ptr + offset, data * 2)
```

### 切片操作
- 禁止 Python 切片 `b[0]` `b[i:j]`
- 单元素: `tl.get_element(tensor, (index,))`
- 切片: `tl.extract_slice(tensor, offsets, sizes, strides)`
- 插入: `tl.insert_slice(full, sub, offsets, sizes, strides)`
- 禁止对 tl.arange 张量用 get_element

### 其他限制
- tl.constexpr 仅在内核参数中使用，host 侧不可用
- 输出张量用 torch.empty / empty_like（避免 zeros/ones 初始化开销）
- 标量转换仅 `scalar.to(type)`，禁止 `tl.float16(scalar)`
- BLOCK_SIZE 必须小于 65536
