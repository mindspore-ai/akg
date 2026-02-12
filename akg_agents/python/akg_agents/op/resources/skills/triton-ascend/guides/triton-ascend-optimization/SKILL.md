---
name: triton-ascend-optimization
description: "Triton Ascend 性能优化、API限制和调试技巧"
level: L3
category: method
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
structure:
  child_skills:
    - triton-ascend-memory
    - triton-ascend-grid-config
    - triton-ascend-debugging
---

# Triton Ascend 性能优化指南

## 1. 性能优化策略

### 1.1 块大小选择

- **原则**: 平衡并行度与资源占用
- **建议**: 1024-2048 for element-wise，根据算子特性调整
- **注意**: BLOCK_SIZE 必须小于 65536

### 1.2 Grid 级别优化

**关键建议**: 将 2D 的 grid 设置修改为 1D 的 grid，之后在内核中进行处理，能够显著降低启动开销。

对于二维输入数据，由于数据是连续存储的，可以将数据处理成（看成）一维形式，统一连续提取、计算，这样可以获得更灵活的切分、操作方式。

### 1.3 核内循环优化

在每个核内进行计算时，对于无需进行 for 循环的场景（例如进行一次搬运就可以完成计算的 vector 类计算），可以通过切分并添加额外 for 循环的手段来隐藏搬运计算开销，编译器会自动将核内 for 循环进行多级流水处理。

同时，可以尝试在核内计算时尝试更大/更小的 Block_size，来平衡并行度和资源占用。

### 1.4 算子拆分策略

对于复杂的算子，例如某些融合算子，在不同计算阶段需要进行不同轴上的 reduce 操作的情况，我们可以将复杂的算子拆开进行处理：

**有的时候一味的融合并不能带来性能收益，反而拆开按顺序单独计算、多次调用不同 kernel 能够带来更好的性能。**

设计按顺序执行的 elementwise 操作和 reduce 操作时，也可以将这部分拆开作为两个 kernel，设置不同的 grid 和 blocksize。

### 1.5 NPU 核心数配置优化

当硬件是 NPU 时，由于物理核数的限制，我们设置核间并行数小于等于虚拟核数时，可能会比设置核间并行数远远超过虚拟核数时性能要更好，因为设置核间并行数远远超过虚拟核数时，编译器自行处理的启动开销会更高。

**核心数配置**:
- **VEC_CORE_NUM**: 向量计算核心数，用于 element-wise、reduce 类算子
- **CUBE_CORE_NUM**: 矩阵计算核心数，用于 matmul、attention 类算子
- 必须在 `__init__` 中获取，避免 forward 中重复调用导致同步开销

### 1.6 NPU 内存访问优化

当硬件是 NPU 时，每次的数据搬运都是以 256Bytes 为单位的，所以在数据读取和存储时，考虑将数据对齐到 256Bytes 的倍数，或许可以提升性能。

数据搬运的带宽性能上限大概是 256*256Bytes，可以参照这个上限来设计数据搬运的策略。

### 1.7 Autotune 优化

可以通过添加 autotune 来优化性能，所以可以在生成时列出多组参数来进行生成，并添加 `@llm_hint("autotune", autotune_configs)` 来提示 LLM 进行优化，列出 autotune_configs 的具体配置。

## 2. 数值稳定性

### 2.1 防溢出处理

**Softmax 数值稳定化**:
```python
# 减去最大值防止 exp 溢出
max_val = tl.max(scores, axis=0)
scores = scores - max_val
p = tl.math.exp2(scores)
```

### 2.2 防负值开方

```python
# 方差计算前确保非负
variance = tl.maximum(variance, 0.0)
std = tl.sqrt(variance + eps)
```

### 2.3 精度提升

- **使用 float32 进行累加**: 即使输入是 float16/bfloat16
- **最后再转换**: 计算完成后再转回目标精度

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# ... 累加计算 ...
result = tl.cast(accumulator, output_dtype)
```

## 3. API 使用限制

### 3.1 禁止使用的语法

**禁止使用**: `return`, `break`, `continue`, `lambda`

Triton 内核是一次性执行完整逻辑，不支持提前返回或跳转语句。

### 3.2 While 循环替代方案

**问题**: Ascend 后端不支持 `while` 循环（运行时动态条件）

**解决方案**: 使用 `for + if` 替代

```python
# 错误：错误：while 循环（Ascend不支持）
@triton.jit
def kernel_while(ptr, n_iters, TILE: tl.constexpr):
    i = 0
    while i < n_iters:  # 运行时动态条件，Ascend不支持
        offset = i * TILE + tl.arange(0, TILE)
        data = tl.load(ptr + offset)
        tl.store(ptr + offset, data * 2)
        i += 1

# 正确：正确：for + if 替代方案
@triton.jit
def kernel_for_if(
    ptr,
    n_iters,              # 运行时动态值
    TILE: tl.constexpr,
    MAX_ITERS: tl.constexpr,  # 编译时常量上界（需足够大）
):
    for i in range(MAX_ITERS):
        if i < n_iters:
            offset = i * TILE + tl.arange(0, TILE)
            data = tl.load(ptr + offset)
            tl.store(ptr + offset, data * 2)
```

**注意事项**:
- `MAX_ITERS` 需设置得足够大，覆盖所有可能的运行时值
- 当实际迭代次数远小于上界时，会有空循环迭代开销

### 3.3 切片操作规范

Triton 不支持 Python 风格的直接切片语法（如 `b[0]` 或 `b[i:j]`），需使用专用 API：

- **单元素提取**: `tl.get_element(tensor, (index,))`
- **切片提取**: `tl.extract_slice(tensor, offsets, sizes, strides)`
- **切片插入**: `tl.insert_slice(full, sub, offsets, sizes, strides)` - 将sub张量插入到ful张量的指定位置

```python
# 一维切片插入
output_sub = x_sub + y_sub
output = tl.insert_slice(output, output_sub, [offset], [size], [1])

# 二维切片插入（逐行构建）
tmp_buf = tl.zeros((rows, cols), dtype)
val = tl.load(in_ptr + offset, mask)
tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, cols), strides=(1, 1))
```

**重要限制**: 禁止对 `tl.arange` 生成的张量使用 `get_element()`
```python
# 错误：错误：offsets = base + tl.arange(0, BLOCK_SIZE); value = tl.get_element(offsets, [i])
# 正确：正确：value = base + i
```

### 3.4 tl.constexpr 正确用法

- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在 host 侧使用**: 启动函数中不可用 tl.constexpr

### 3.5 输出张量创建规范

- 正确：使用 `torch.empty` 或 `torch.empty_like`
- 错误：避免 `torch.zeros` 或 `torch.ones`（避免不必要的初始化开销）

### 3.6 Ascend 后端特殊限制

- **避免使用 tl.where 计算内存偏移**: Ascend 后端对 `tl.where` 生成的复杂指针运算支持不完全
- **标量类型转换**: 仅支持 `scalar.to(type)`，禁止使用 `tl.float16(scalar)`
- **BLOCK_SIZE 限制**: 必须小于 65536

### 3.7 Grid 设置规范

- **维度限制**: grid 必须是 tuple 类型，最多 3 维
- **大小限制**: 各维度乘积不超过 65535