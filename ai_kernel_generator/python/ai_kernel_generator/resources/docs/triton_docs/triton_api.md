# Triton 核心 API 参考

## 编程模型 (Programming Model)

*定义 Triton 编程模型的核心概念，包括张量、程序ID、张量描述符等基础构建块*

### 程序标识和网格
- `tl.program_id(axis)`: 获取当前程序在指定轴上的 ID
  - `axis`: 3D 启动网格的轴，必须是 0、1 或 2
- `tl.num_programs(axis)`: 获取指定轴上的程序总数
  - `axis`: 3D 启动网格的轴，必须是 0、1 或 2

### 张量和张量描述符
- `tl.tensor(...)`: 表示 N 维数值或指针数组
- `tl.tensor_descriptor(...)`: 表示全局内存中张量的描述符
- `tl.make_tensor_descriptor(...)`: 创建张量描述符对象
- `tl.load_tensor_descriptor(...)`: 从张量描述符加载数据块
- `tl.store_tensor_descriptor(...)`: 向张量描述符存储数据块

---

## 张量创建 (Creation Ops)

*用于创建和初始化张量的函数，支持各种数据类型和填充方式*

### 基础创建函数
- `tl.arange(start, end, step=1, dtype=tl.int32)`: 创建范围数组
  - `start`: 区间起始值，必须是2的幂
  - `end`: 区间结束值，必须是大于start的2的幂
- `tl.zeros(shape, dtype)`: 创建零张量
- `tl.zeros_like(input)`: 创建与输入张量形状和类型相同的零张量
- `tl.full(shape, value, dtype)`: 创建填充指定值的张量

### 类型转换和广播
- `tl.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`: 类型转换
- `tl.broadcast(input, other)`: 广播两个张量到兼容形状
- `tl.broadcast_to(input, shape)`: 将张量广播到指定形状

### 张量连接
- `tl.cat(input, other)`: 连接两个张量

---

## 形状操作 (Shape Manipulation)

*改变张量形状、维度和布局的操作，如重塑、转置、分割等*

### 维度操作
- `tl.expand_dims(input, axis)`: 插入长度为1的新维度
- `tl.permute(input, dims)`: 置换张量维度
- `tl.trans(tensor)`: 转置张量
- `tl.reshape(input, shape)`: 重塑张量形状
- `tl.view(input, shape)`: 返回不同形状的张量视图

### 张量拆分和组合
- `tl.split(input)`: 沿最后维度分割张量（维度大小必须为2）
- `tl.join(a, b)`: 在新的次要维度连接张量
- `tl.interleave(a, b)`: 沿最后维度交错两个张量

---

## 线性代数 (Linear Algebra)

*线性代数运算，主要用于矩阵乘法和相关计算*

### 矩阵运算
- `tl.dot(a, b, acc=None, allow_tf32=True)`: 计算两个张量的矩阵乘积，可选累加器
- `tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format)`: 微缩放矩阵乘法

---

## 内存操作 (Memory/Pointer)

*内存访问和指针操作，用于高效的数据加载和存储*

### 基础内存操作
- `tl.load(pointer, mask=None, other=None, boundary_check=None, eviction_policy="")`: 从指针地址加载数据
  - `pointer`: 指针数组或单个指针
  - `mask`: 布尔掩码，控制哪些元素被加载
  - `other`: 当 mask 为 False 时使用的默认值
  - `boundary_check`: 边界检查的维度元组
  - `eviction_policy`: 驱逐策略
- `tl.store(pointer, value, mask=None, boundary_check=None, eviction_policy="")`: 向指针地址存储数据
  - `pointer`: 指针数组或单个指针
  - `value`: 要存储的值
  - `mask`: 布尔掩码，控制哪些元素被存储
  - `boundary_check`: 边界检查的维度元组
  - `eviction_policy`: 驱逐策略

### 高级内存操作
- `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`: 创建块指针，用于高效的 2D 数据访问
- `tl.advance(block_ptr, offsets)`: 推进块指针位置

---

## 索引操作 (Indexing)

*索引和选择操作，用于访问张量的特定元素或区域*

### 索引函数
- `tl.where(condition, x, y)`: 根据条件从 x 或 y 中选择元素
- `tl.flip(x, dim)`: 沿指定维度翻转张量
- `tl.swizzle2d(i, j, size_i, size_j, size_g)`: 2D索引变换

---

## 数学函数 (Math Functions)

*数学函数库，包括算术运算、三角函数、指数对数等*

### 基础数学函数
- `tl.abs(x)`: 计算元素级绝对值
- `tl.ceil(x)`: 计算元素级向上取整
- `tl.floor(x)`: 计算元素级向下取整
- `tl.clamp(x, min, max)`: 将输入张量限制在 [min, max] 范围内

### 指数和对数函数
- `tl.exp(x)`: 计算元素级指数函数 e^x
- `tl.exp2(x)`: 计算元素级以2为底的指数函数
- `tl.log(x)`: 计算元素级自然对数
- `tl.log2(x)`: 计算元素级以2为底的对数

### 平方根函数
- `tl.sqrt(x)`: 计算元素级快速平方根
- `tl.sqrt_rn(x)`: 计算元素级精确平方根（IEEE标准舍入）
- `tl.rsqrt(x)`: 计算元素级倒数平方根

### 三角函数
- `tl.cos(x)`: 计算元素级余弦函数
- `tl.sin(x)`: 计算元素级正弦函数

### 激活函数
- `tl.sigmoid(x)`: 计算元素级sigmoid函数
- `tl.softmax(x, axis)`: 计算元素级softmax函数

### 除法函数
- `tl.fdiv(x, y)`: 计算元素级快速除法
- `tl.div_rn(x, y)`: 计算元素级精确除法（IEEE标准舍入）
- `tl.cdiv(x, div)`: 计算向上取整除法

### 比较函数
- `tl.maximum(x, y, propagate_nan=None)`: 计算元素级最大值
- `tl.minimum(x, y, propagate_nan=None)`: 计算元素级最小值

### 特殊函数
- `tl.erf(x)`: 计算元素级误差函数
- `tl.fma(x, y, z)`: 计算元素级融合乘加运算 (x*y+z)
- `tl.umulhi(x, y)`: 计算 2N 位乘积的高 N 位

---

## 归约操作 (Reduction)

*归约操作，在指定维度上进行求和、最大值、最小值等聚合计算*

### 基础归约
- `tl.sum(input, axis=None, keep_dims=False, dtype=None)`: 求和归约
- `tl.max(input, axis=None, keep_dims=False)`: 最大值归约
- `tl.min(input, axis=None, keep_dims=False)`: 最小值归约
- `tl.reduce(input, axis, combine_fn, keep_dims=False)`: 自定义归约函数
- `tl.xor_sum(input, axis, keep_dims=False)`: 异或和

### 索引归约
- `tl.argmax(input, axis=None, keep_dims=False, tie_break_left=True)`: 最大值索引
- `tl.argmin(input, axis=None, keep_dims=False, tie_break_left=True)`: 最小值索引

### 累积操作
- `tl.cumsum(input, axis, reverse=False, dtype=None)`: 累积和

### 编译器提示
- `tl.assume(condition)`: 允许编译器假设条件为真

---

## 扫描排序 (Scan/Sort)

*扫描和排序算法，用于前缀和、累积运算和数据排序*

### 扫描操作
- `tl.associative_scan(input, axis, combine_fn, reverse=False)`: 结合扫描
- `tl.cumprod(input, axis, reverse=False)`: 累积乘积

### 数据操作
- `tl.gather(src, index, axis=0)`: 按索引收集数据
- `tl.histogram(input, num_bins, mask=None)`: 计算直方图

---

## 原子操作 (Atomic)

*原子操作，保证多线程环境下的内存操作安全性*

### 算术原子操作
- `tl.atomic_add(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子加法
- `tl.atomic_max(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子最大值
- `tl.atomic_min(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子最小值

### 逻辑原子操作
- `tl.atomic_and(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子逻辑与
- `tl.atomic_or(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子逻辑或
- `tl.atomic_xor(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子逻辑异或

### 交换原子操作
- `tl.atomic_xchg(pointer, val, mask=None, sem='acq_rel', scope="")`: 原子交换
- `tl.atomic_cas(pointer, cmp, val, mask=None, sem='acq_rel', scope="")`: 原子比较交换

---

## 随机数生成 (Random)

*随机数生成器，用于生成各种分布的随机数*

### 随机数函数
- `tl.rand(seed, offset, n_rounds=4)`: 均匀分布随机数
- `tl.randn(seed, offset, n_rounds=4)`: 正态分布随机数
- `tl.randint(seed, offset, n_rounds=4)`: 随机整数
- `tl.randint4x(seed, offset, n_rounds=4)`: 四个随机整数

---

## 迭代器 (Iterators)

*循环迭代器，用于控制程序的循环结构*

### 循环控制
- `tl.range(start, end, step=1)`: 动态范围迭代器
- `tl.static_range(start, end, step=1)`: 静态范围迭代器（编译时展开）

---

## 内联汇编 (Inline Assembly)

*内联汇编接口，允许直接嵌入底层代码*

### 汇编函数
- `tl.inline_asm_elementwise(asm_string, inputs, outputs, constraints)`: 在张量上执行内联汇编

---

## 编译器提示 (Compiler Hints)

*编译器优化提示，帮助编译器生成更高效的代码*

### 优化提示
- `tl.multiple_of(input, value)`: 标记值是某个数的倍数（优化提示）
- `tl.max_contiguous(input, value)`: 标记连续内存访问
- `tl.max_constancy(input, value)`: 标记常量值
- `tl.debug_barrier()`: 插入同步屏障

---

## 调试工具 (Debug)

*调试工具，用于程序调试、断言和运行时检查*

### 运行时调试
- `tl.device_print(prefix, *args)`: 设备端运行时打印（需要TRITON_DEBUG=1）
- `tl.device_assert(condition, msg='')`: 设备端运行时断言（需要TRITON_DEBUG=1）

### 编译时调试
- `tl.static_print(*args)`: 编译时打印
- `tl.static_assert(condition, msg='')`: 编译时断言

---

## 内存访问模式

### 连续访问模式
```python
offsets = tl.arange(0, BLOCK_SIZE)
ptrs = base_ptr + offsets
data = tl.load(ptrs, mask=offsets < n_elements)
```

### 步幅访问模式
```python
offsets = tl.arange(0, BLOCK_SIZE)
ptrs = base_ptr + offsets * stride
data = tl.load(ptrs, mask=offsets < n_elements)
```

### 2D 访问模式
```python
row_offsets = tl.arange(0, BLOCK_M)[:, None]
col_offsets = tl.arange(0, BLOCK_N)[None, :]
ptrs = base_ptr + row_offsets * stride + col_offsets
```

### 块指针访问（推荐用于矩阵操作）
```python
# 创建块指针
block_ptr = tl.make_block_ptr(
    base=ptr,
    shape=(M, N),
    strides=(stride_m, stride_n),
    offsets=(start_m, start_n),
    block_shape=(BLOCK_M, BLOCK_N),
    order=(1, 0)  # 行优先
)

# 加载数据
data = tl.load(block_ptr)

# 推进指针
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```

---

## 控制流

### 循环结构
```python
# 标准 Python for 循环
for i in range(start, end, step):
    # 循环体

# Triton 优化的循环
for i in tl.range(start, end, step):
    # 循环体，可能有更好的优化

# 静态循环（编译时展开）
for i in tl.static_range(start, end, step):
    # 循环体，编译时展开
```

### 条件语句
```python
# 标准条件语句
if condition:
    # 条件为真的代码块
else:
    # 条件为假的代码块

# 条件选择函数
result = tl.where(condition, value_if_true, value_if_false)
```

---

## 常用工具函数

### 网格计算
```python
# 除法向上取整
triton.cdiv(a, b)  # 等价于 (a + b - 1) // b

# 下一个2的幂
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

### 网格启动
```python
# 1D 网格
grid = (n_programs,)
kernel[grid](args...)

# 2D 网格
grid = (n_programs_x, n_programs_y)
kernel[grid](args...)

# 使用 lambda 动态计算网格大小
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
kernel[grid](args..., BLOCK_SIZE=block_size)
```

---

## 性能优化提示

### 内存合并
- 确保连续内存访问以提高带宽利用率
- 使用适当的块大小（通常是 2 的幂）
- 使用块指针进行高效的 2D 内存访问

### 掩码优化
- 尽量减少掩码的使用以提高性能
- 当必须使用掩码时，为 `other` 参数选择合适的值

### 编译器提示优化
```python
# 标记值是某个数的倍数，帮助编译器优化
value = tl.multiple_of(value, 32)

# 标记连续访问模式
ptr = tl.max_contiguous(ptr, BLOCK_SIZE)

# 编译时断言，确保条件成立
tl.static_assert(BLOCK_SIZE <= MAX_BLOCK_SIZE)
```

---

## 编译时常量和装饰器

- `@triton.jit`: 装饰器，用于标记 Triton 内核函数
- `tl.constexpr`: 编译时常量类型注解
- `tl.static_assert(condition)`: 编译时断言

---
