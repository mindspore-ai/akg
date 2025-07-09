# Triton 核心 API & 编程指南 (大模型精简版)

本文档为大模型提供 Triton 编程的核心要点，旨在高效、正确地生成 Triton 内核代码。

## 1. 核心概念与网格

- **内核 (Kernel)**: 使用 `@triton.jit` 装饰的 Python 函数。
- **网格 (Grid)**: 内核在多维度网格上并行执行。通过 `pid` 区分不同程序。
  - `pid = tl.program_id(axis)`: 获取当前程序ID (axis: 0, 1, or 2)。
  - `num_pids = tl.num_programs(axis)`: 获取该轴上的总程序数。
- **启动网格**: `kernel[grid](...)`。`grid` 通常是元组或 lambda 表达式。
  - **工具**: `triton.cdiv(a, b)` 是计算网格大小的常用工具 (向上取整除法)。
  - **示例**: `grid = (triton.cdiv(N, BLOCK_SIZE), )`

## 2. 内存操作 (最关键)

Triton 的性能核心在于高效的内存访问。

- **指针**: 指针可直接进行算术运算来寻址，如 `ptr + offset`。
- **加载/存储**:
  - `tl.load(pointer, mask=None, other=None, boundary_check=None)`: 带掩码加载。
  - `tl.store(pointer, value, mask=None, boundary_check=None)`: 带掩码存储。
  - **`mask` 至关重要**: 用于防止越界访问，尤其在处理不规整的张量末尾时。
- **块指针 (2D数据高效访问)**: 强烈推荐用于矩阵等二维数据块的访问。
  - `tl.make_block_ptr(...)`: 创建块指针。
  - `tl.advance(ptr, offsets)`: 移动块指针。
  - **代码范式**:
    ```python
    # 1. 创建块指针
    block_ptr = tl.make_block_ptr(
        base=ptr,                # 基础指针
        shape=(M, N),            # 完整矩阵形状
        strides=(stride_m, stride_n), # 步长
        offsets=(start_m, start_n),   # 当前块的偏移
        block_shape=(BLOCK_M, BLOCK_N), # 块的形状
        order=(1, 0)             # 行主序
    )
    # 2. 加载数据 (带边界检查)
    data = tl.load(block_ptr, boundary_check=(0, 1))
    # 3. 移动指针到下一个块
    block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
    ```

## 3. 张量、计算与归约

- **创建**: `tl.arange(start, end)`, `tl.zeros(shape, dtype)`, `tl.full(shape, value, dtype)`.
- **形状变换**: `tl.reshape`, `tl.trans`, `tl.permute`, `tl.expand_dims`, `tl.broadcast_to`.
- **类型转换**: `tl.cast(input, dtype)`.
- **线性代数**: `tl.dot(a, b, acc=None, allow_tf32=True)` 是矩阵乘法的核心。
- **逐元素数学**: `tl.exp`, `tl.log`, `tl.sqrt`, `tl.cos`, `tl.abs`, `tl.maximum`, `tl.minimum`。
- **条件选择**: `tl.where(condition, x, y)` (动态选择，对性能友好)。
- **归约**: `tl.sum(x, axis)`, `tl.max(x, axis)`, `tl.min(x, axis)`, `tl.reduce(x, axis, combine_fn)`.
- **原子操作**: 在并行写入全局内存时保证线程安全。
  - `tl.atomic_add`, `tl.atomic_max`, `tl.atomic_min`, `tl.atomic_cas`.

## 4. 控制流

### 重要约束
- **绝对不能使用 `return` 语句**：Triton 内核不支持早期返回
- **绝对不能使用 `break/continue` 语句**：这些控制流语句不受支持
- **边界检查必须用 `mask`**：所有越界处理都通过 `mask` 参数实现

### 支持的控制流
- **静态条件**: Python 的 `if` 语句用于**编译时常量**判断，无运行时开销。
  - `if BLOCK_SIZE > 64:`
- **动态条件**: `tl.where` 用于**张量**的条件判断，是 SIMD 友好的。
- **循环**:
  - **Python `for` 循环**: Triton 会尽力优化。
  - **`tl.static_range(start, end)`**: 编译时展开循环，用于循环次数少且固定的情况。

### 正确的边界处理模式
```python
# 错误：使用return
if pid >= max_pids:
    return  # 不支持！

# 正确：使用mask
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask)
```

## 5. 性能与编译器提示

- **内存合并**: 核心优化点。确保 `tl.load`/`tl.store` 访问的是**连续的内存地址**。块大小通常为2的幂。
- **使用块指针**: 对于2D数据，优先使用 `tl.make_block_ptr`，它能更好地处理内存合并和边界问题。
- **编译器提示**:
  - `tl.static_assert(condition, msg)`: 编译时断言，用于检查约束。
  - `tl.constexpr`: 标记编译时常量。 