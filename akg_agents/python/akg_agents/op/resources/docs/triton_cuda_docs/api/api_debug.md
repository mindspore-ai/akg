# Triton API 速查手册

## 1. 编译与执行
- **`@triton.jit`**: 编译 Python 函数为硬件内核。*注：禁止 return/break/continue。*
- **`tl.program_id(axis)`**: 获取当前程序在该轴的 ID。
- **`tl.num_programs(axis)`**: 获取该轴的总程序数。
- **`triton.cdiv(a, b)`**: 向上取整除法 (a+b-1)//b，常用于网格计算。

## 2. 内存读取与指针
- **`tl.load(ptr, mask, other)`**: 加载数据。可配合 mask 处理边界。
- **`tl.store(ptr, val, mask)`**: 存储数据到全局内存。
- **`tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`**: 创建高效的 2D 块指针。
- **`tl.advance(ptr, offsets)`**: 移动块指针。

## 3. 张量创建与转换
- **`tl.arange(start, end)`**: 创建连续整数索引序列。
- **`tl.zeros/full(shape, value, dtype)`**: 创建全零或指定值的张量。
- **`tl.cast(x, dtype)`**: 显式类型转换。
- **`tl.where(cond, x, y)`**: SIMD 友好的条件选择。

## 4. 数学运算与归约
- **`tl.cdiv(a, b)`**: kernel 内部的向上取整除法 ⌈a/b⌉。
- **`tl.dot(a, b, acc)`**: 核心矩阵乘法 (MatMul)。
- **`tl.sum/max(x, axis)`**: 沿轴归约。
- **`tl.cumsum/cumprod(x, axis)`**: 沿轴计算累积和/积。

## 5. 原子操作与常量
- **`tl.atomic_add/max(ptr, val)`**: 线程安全的操作。
- **`tl.constexpr`**: 标记编译器常量。

## 6. 块分配优化 API
- **`tl.swizzle2d(i, j, size_i, size_j, group_size)`**: 2D 块重排，提升缓存局部性。