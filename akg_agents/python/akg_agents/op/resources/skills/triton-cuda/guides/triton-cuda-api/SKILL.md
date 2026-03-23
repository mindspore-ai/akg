---
name: triton-cuda-api
description: "Triton CUDA API 完整参考手册，包括 tl.load/store、tl.reduce、tl.dot、tl.atomic 等核心函数的签名、参数和使用示例。适用于需要查阅具体 API 用法、了解函数参数含义的任意 Triton CUDA 内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Triton CUDA API 参考手册

本文档提供 Triton 核心 API 的详细参考，包括函数签名、参数说明和使用示例。

## 1. 内核装饰器

### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **作用**: 将 Python 函数编译为 GPU 内核
- **约束**: 函数内部不能使用 `return`、`break`、`continue` 语句

## 2. 程序 ID 与网格 API

### tl.program_id(axis)
```python
pid = tl.program_id(axis)  # axis: 0, 1, or 2
```
- **参数**: `axis` - 维度轴 (0, 1, 2)
- **返回**: 当前程序在该轴上的 ID
- **用途**: 确定当前程序块处理的数据范围

### tl.num_programs(axis)
```python
num_pids = tl.num_programs(axis)  # axis: 0, 1, or 2
```
- **参数**: `axis` - 维度轴 (0, 1, 2)
- **返回**: 该轴上的总程序数
- **用途**: 计算网格大小和边界条件

### triton.cdiv(a, b)
```python
grid_size = triton.cdiv(total_elements, block_size)
```
- **参数**: `a`, `b` - 被除数和除数
- **返回**: 向上取整的除法结果
- **用途**: host 侧使用，计算启动网格大小

## 3. 内存操作 API

### tl.load(pointer, mask=None, other=None, boundary_check=None)
```python
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```
- **参数**:
  - `pointer`: 内存指针
  - `mask`: 布尔掩码，True 表示有效位置
  - `other`: 掩码为 False 时的默认值
  - `boundary_check`: 边界检查维度 (0, 1) 或 None
- **返回**: 加载的张量数据
- **用途**: 从全局内存加载数据

### tl.store(pointer, value, mask=None, boundary_check=None)
```python
tl.store(ptr + offsets, result, mask=mask)
```
- **参数**:
  - `pointer`: 内存指针
  - `value`: 要存储的值
  - `mask`: 布尔掩码，True 表示有效位置
  - `boundary_check`: 边界检查维度 (0, 1) 或 None
- **用途**: 将数据存储到全局内存

### tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
```python
block_ptr = tl.make_block_ptr(
    base=ptr,                    # 基础指针
    shape=(M, N),                # 完整矩阵形状
    strides=(stride_m, stride_n), # 步长
    offsets=(start_m, start_n),   # 当前块偏移
    block_shape=(BLOCK_M, BLOCK_N), # 块形状
    order=(1, 0)                 # 内存布局顺序
)
```
- **参数**:
  - `base`: 基础内存指针
  - `shape`: 完整张量的形状
  - `strides`: 每个维度的步长
  - `offsets`: 当前块的起始偏移
  - `block_shape`: 当前块的大小
  - `order`: 内存布局顺序 (1, 0) 表示行主序
- **返回**: 块指针对象
- **用途**: 高效访问 2D 数据块

### tl.advance(ptr, offsets)
```python
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```
- **参数**:
  - `ptr`: 块指针
  - `offsets`: 各维度的偏移量
- **返回**: 移动后的块指针
- **用途**: 移动块指针到下一个位置

## 4. 张量创建与操作 API

### tl.arange(start, end)
```python
offsets = tl.arange(0, BLOCK_SIZE)
```
- **参数**: `start`, `end` - 起始和结束值
- **返回**: 连续整数序列
- **用途**: 创建索引序列

### tl.zeros(shape, dtype)
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `dtype`: 数据类型
- **返回**: 全零张量

### tl.full(shape, value, dtype)
```python
ones = tl.full((M, N), 1.0, dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `value`: 填充值
  - `dtype`: 数据类型
- **返回**: 填充指定值的张量

### tl.cast(input, dtype)
```python
float_data = tl.cast(int_data, tl.float32)
```
- **参数**:
  - `input`: 输入张量
  - `dtype`: 目标数据类型
- **返回**: 类型转换后的张量

## 5. 数学运算 API

### tl.dot(a, b, acc=None, allow_tf32=True)
```python
result = tl.dot(a, b, acc=accumulator)
```
- **参数**:
  - `a`, `b`: 输入矩阵
  - `acc`: 累加器 (可选)
  - `allow_tf32`: 是否允许 TF32 精度（CUDA 特有，Ampere+ GPU）
- **返回**: 矩阵乘法结果
- **用途**: 核心矩阵乘法操作，利用 Tensor Core 加速

### tl.sum(x, axis)
```python
block_sum = tl.sum(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 归约结果

### tl.max(x, axis)
```python
max_val = tl.max(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 最大值

### tl.min(x, axis)
```python
min_val = tl.min(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 最小值

### tl.where(condition, x, y)
```python
result = tl.where(mask, data, 0.0)
```
- **参数**:
  - `condition`: 条件张量
  - `x`, `y`: 选择值
- **返回**: 根据条件选择的值
- **用途**: SIMD 友好的条件选择

### tl.exp(x) / tl.log(x) / tl.sqrt(x)
```python
exp_val = tl.exp(x)
log_val = tl.log(x)
sqrt_val = tl.sqrt(x)
```
- **用途**: 基本数学函数，CUDA 后端直接支持

### tl.sigmoid(x)
```python
sigmoid_val = tl.sigmoid(x)
```
- **用途**: Sigmoid 激活函数

### tl.extra.cuda.libdevice.tanh(x)
```python
tanh_val = tl.extra.cuda.libdevice.tanh(x)
```
- **用途**: 双曲正切函数
- **注意**: CUDA 后端无 `tl.tanh` 或 `tl.math.tanh`，须使用 `tl.extra.cuda.libdevice.tanh`

### tl.math.exp2(x) / tl.math.log2(x)
```python
exp2_val = tl.math.exp2(x)
log2_val = tl.math.log2(x)
```
- **用途**: 以 2 为底的指数/对数运算

### tl.cumsum(input, axis=0, reverse=False, dtype=None)
```python
cumulative_sum = tl.cumsum(data, axis=0)
reverse_cumsum = tl.cumsum(data, axis=1, reverse=True)
```
- **参数**:
  - `input`: 输入张量
  - `axis`: 累积求和的轴 (默认为 0)
  - `reverse`: 是否反向累积 (默认为 False)
  - `dtype`: 输出数据类型 (可选，默认与输入相同)
- **返回**: 累积求和结果张量
- **用途**: 计算沿指定轴的累积和，常用于前缀和计算

### tl.cumprod(input, axis=0, reverse=False)
```python
cumulative_prod = tl.cumprod(data, axis=0)
```
- **参数**:
  - `input`: 输入张量
  - `axis`: 累积乘积的轴 (默认为 0)
  - `reverse`: 是否反向累积 (默认为 False)
- **返回**: 累积乘积结果张量

## 6. 原子操作 API

### tl.atomic_add(pointer, value)
```python
tl.atomic_add(output_ptr, block_sum)
```
- **参数**:
  - `pointer`: 目标内存指针
  - `value`: 要添加的值
- **用途**: 线程安全的加法操作

### tl.atomic_max(pointer, value)
```python
tl.atomic_max(max_ptr, local_max)
```
- **参数**:
  - `pointer`: 目标内存指针
  - `value`: 要比较的值
- **用途**: 线程安全的最大值更新

### tl.atomic_min(pointer, value)
```python
tl.atomic_min(min_ptr, local_min)
```
- **参数**:
  - `pointer`: 目标内存指针
  - `value`: 要比较的值
- **用途**: 线程安全的最小值更新

### tl.atomic_cas(pointer, cmp, val)
```python
old = tl.atomic_cas(ptr, expected, desired)
```
- **参数**:
  - `pointer`: 目标内存指针
  - `cmp`: 期望值
  - `val`: 新值
- **返回**: 原始值
- **用途**: 比较并交换操作

### tl.constexpr
```python
BLOCK_SIZE: tl.constexpr = 1024
```
- **用途**: 标记编译时常量参数
- **约束**: 必须在函数签名中声明

## 7. 块分配优化 API

### tl.swizzle2d(i, j, size_i, size_j, group_size)
```python
task_i, task_j = tl.swizzle2d(block_i, block_j, NUM_BLOCKS_I, NUM_BLOCKS_J, GROUP_SIZE)
```
- **参数**:
  - `i`, `j`: 原始块索引
  - `size_i`, `size_j`: 总块数
  - `group_size`: 分组大小(通常为 2/4/8)
- **返回**: 重排后的块索引 (task_i, task_j)
- **用途**: 2D 块重排，提升 L2 缓存局部性
- **适用场景**: 矩阵乘法等多维块计算，改善数据复用

## 使用建议

1. **内存操作**: 优先使用 `tl.make_block_ptr` 处理 2D 数据
2. **边界检查**: 始终使用 `mask` 或 `boundary_check` 防止越界
3. **原子操作**: 仅在必要时使用，有性能开销
4. **数据类型**: 注意类型转换，使用 `tl.cast` 显式转换
5. **编译优化**: 使用 `tl.constexpr` 标记常量参数
6. **Tensor Core**: MatMul 类操作使用 `allow_tf32=True` 利用 Tensor Core
