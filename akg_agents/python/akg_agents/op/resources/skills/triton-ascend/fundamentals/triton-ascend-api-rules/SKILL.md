---
name: triton-ascend-api-rules
description: "Triton Ascend hard API restrictions and forbidden syntax. MUST-follow rules that apply to every kernel: forbidden control flow (return/break/continue/lambda/while), tensor slice/index restrictions, scalar conversion rules, BLOCK_SIZE upper bound. Violating any of these produces a compile or runtime error on Ascend."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Triton Ascend API Hard Rules (MUST follow)

This file lists rules whose violation causes a compile or runtime error
on Ascend. They apply to every kernel in this DSL — no exceptions.

## 禁止使用的语法

- `return` / `break` / `continue` → 使用 mask 控制
- lambda → 内联函数或 tl.where
- 链式布尔运算 → 分步计算 mask
- 张量直接索引 → tl.load / tl.store
- if-else 中负偏移 → tl.maximum(offset, 0)
- Ascend: 复杂 tl.where → if-else
- Ascend: while 循环 → for 替代
- Ascend: range() 的 start/stop 混用运行时变量和 constexpr → 用全 constexpr 的 range + 循环体内运行时 if 跳过

## While 循环替代（Ascend）

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

## 切片操作

- 禁止 Python 切片 `b[0]` `b[i:j]`
- 单元素: `tl.get_element(tensor, (index,))`
- 切片: `tl.extract_slice(tensor, offsets, sizes, strides)`
- 插入: `tl.insert_slice(full, sub, offsets, sizes, strides)`
- 禁止对 tl.arange 张量用 get_element

## 其他限制

- tl.constexpr 仅在内核参数中使用，host 侧不可用
- 输出张量用 torch.empty / empty_like（避免 zeros/ones 初始化开销）
- 标量转换仅 `scalar.to(type)`，禁止 `tl.float16(scalar)`
- BLOCK_SIZE 必须小于 65536
