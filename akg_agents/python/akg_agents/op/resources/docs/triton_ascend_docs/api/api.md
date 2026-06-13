# Triton Ascend Core API 参考手册

本文档是 Coder 默认展示的 Triton Ascend core API 文档。这里列出的 API 会被视为“已展示 API”；Triton API database 召回到同名 API 时只补充候选关系，不重复展开详细文档。低频 API 不在本文档常驻展示，应通过 database recall 按任务补充。

## 0. 高频 API 使用约束

- `tl.arange(0, BLOCK_*)` 的逻辑长度应使用编译期常量，通常取 2 的幂；真实尾部用 mask 处理，不要把动态 runtime shape 直接作为 `end`。
- `tl.load` / `tl.store` 的 pointer、mask、value shape 必须可广播到同一 block shape；mask 必须在内存访问前排除越界地址。
- Ascend grid 必须是 tuple 且最多 3 维，总 program 数不要超过 `65535`；大 shape 优先使用固定核心数 grid-stride 或连续分块，而不是为每个 tile 启动一个 program。
- `tl.dot` 适合矩阵/批矩阵 contraction，输入通常为 `(BLOCK_M, BLOCK_K)` 和 `(BLOCK_K, BLOCK_N)`；不要用 `A * B` 冒充矩阵乘。
- Ascend `tl.dot` 不要传 `allow_tf32` 或 `input_precision`；这些是 CUDA 精度控制语义，Ascend 后端不支持。
- `tl.zeros`、`tl.full`、`tl.reshape`、`tl.static_range` 等涉及 shape 或循环边界的参数必须是编译期可确定的 literal 或 `tl.constexpr` meta 参数。
- 动态循环优先使用 `tl.range`；只有小的编译期常量循环才使用 `tl.static_range`，不要在 kernel 中用依赖 runtime 参数的 Python `range(start, end, step)`。
- 不要生成或调优 CUDA-only launch 参数，例如 `num_warps`、`num_ctas`、`num_stages`、`num_buffers_warp_spec`、`num_consumer_groups`、`reg_dec_producer`、`reg_inc_consumer`、`maxnreg`。
- Triton language 中不要臆造 API；常见不存在的写法包括 `tl.div`、`tl.mod`、`tl.prod`。整数除法/取模优先用 `//`、`%`，乘积在 host 侧或静态表达式中展开。

## 1. 内核装饰器

### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **作用**: 将 Python 函数编译为硬件内核
- **约束**:
  - kernel 内部只能使用 Triton 支持的 Python 子集。
  - 函数内部不要使用 `return`、`break`、`continue` 语句。
  - Python tuple/list 不要作为 runtime 对象在 kernel 内参与 `%`、`//`、比较或广播；stride、padding、dilation 等应拆成标量 meta 参数或 runtime 标量。
  - 需要作为编译期常量的 shape、tile、循环上界应在函数签名中声明为 `tl.constexpr`。
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
- **用途**: host侧使用，计算启动网格大小

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
- **约束**:
  - `pointer` 和 `mask` 的 block shape 必须兼容。例如 pointer 为 `(BLOCK_M, BLOCK_K)` 时，mask 也应能广播到 `(BLOCK_M, BLOCK_K)`。
  - 对 padding、tail、反推输入越界等情况，必须用 `mask` 阻止越界地址参与 load；不要先越界 load 再用 `tl.where` 修正数值。
  - `other` 应匹配归约语义：sum/dot 用 `0.0`，min 用 `float("inf")`，max 用 `-float("inf")`。
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
- **约束**:
  - `pointer`、`value`、`mask` 的 block shape 必须兼容。
  - store mask 必须覆盖所有输出 tail 维度，避免多个 program 写同一个输出元素。
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
- **约束**:
  - `start` 和 `end` 应是编译期可确定值，`end - start` 通常需要是 2 的幂。
  - 对非 2 的幂或动态真实长度，使用 padded `BLOCK_SIZE` 创建 offsets，再用 `offsets < logical_size` 做 mask。
  - 不要写 `tl.arange(0, N)`，其中 `N` 是普通 runtime 参数。
### tl.zeros(shape, dtype)
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `dtype`: 数据类型
- **返回**: 全零张量
- **约束**: `shape` 必须是编译期可确定 tuple；`BLOCK_M/BLOCK_N` 应为 `tl.constexpr` 或 literal。
### tl.full(shape, value, dtype)
```python
ones = tl.full((M, N), 1.0, dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `value`: 填充值
  - `dtype`: 数据类型
- **返回**: 填充指定值的张量
- **约束**: `shape` 必须是编译期可确定 tuple；用于 reduction identity 时要选择正确初值。
### tl.reshape(input, shape)
```python
y = tl.reshape(x, (BLOCK_M, BLOCK_N))
```
- **参数**:
  - `input`: 输入 block tensor
  - `shape`: 目标形状
- **返回**: reshape 后的 block tensor
- **约束**:
  - `shape` 必须是编译期可确定 tuple，元素总数需与输入一致。
  - 不要用 `tl.reshape` 处理 host 侧 tensor 布局转换；host tensor 的 `.reshape/.view/.contiguous` 应在 `forward()` 中完成。
### tl.expand_dims(input, axis)
```python
x_col = tl.expand_dims(x, 1)
```
- **参数**:
  - `input`: 输入 block tensor
  - `axis`: 新增维度位置
- **用途**: 显式构造广播维度，例如把 `(BLOCK_M,)` 变成 `(BLOCK_M, 1)`。
- **约束**:
  - `axis` 应是编译期可确定的整数。
  - 也可以使用 `x[:, None]` / `x[None, :]` 形成广播维度，但不要使用不支持的复杂 Python tensor indexing。
### tl.cast(input, dtype)
```python
float_data = tl.cast(int_data, tl.float32)
```
- **参数**:
  - `input`: 输入张量
  - `dtype`: 目标数据类型
- **返回**: 类型转换后的张量

## 5. 数学运算 API

### tl.cdiv(a, b)
```python
result = tl.cdiv(offset, BLOCK_SIZE)
```
- **参数**: `a`, `b` - 被除数和除数
- **返回**: 向上取整的除法结果 ⌈a/b⌉
- **用途**: kernel内部使用，计算向上整除结果，等价于 `(a + b - 1) // b`

### tl.dot(a, b, acc=None)
```python
result = tl.dot(a, b, acc=accumulator)
```
- **参数**:
  - `a`, `b`: 输入矩阵
  - `acc`: 累加器 (可选)
- **返回**: 矩阵乘法结果
- **用途**: 核心矩阵乘法操作
- **Ascend 约束**:
  - 不要传 `allow_tf32` 或 `input_precision`；这些是 CUDA 精度控制语义，Ascend 后端不支持。
  - `a` 和 `b` 通常应为 rank-2 block tensor，形状分别为 `(BLOCK_M, BLOCK_K)` 和 `(BLOCK_K, BLOCK_N)`；rank-3 表示 batched matmul。
  - K 维必须一致。真实 `K_total` 不是 tile 整数倍时，用 padded `BLOCK_K` 和 `offs_k < K_total` mask，masked load 的 `other` 使用 `0.0`。
  - `BLOCK_K` 通常取 cube/tensor-core 友好的大小，常见至少 16；不要使用 `(1, BLOCK_K) @ (BLOCK_K, 1)` 伪装矩阵化。
  - `acc += A * B` 不能替代 `tl.dot(A, B)`，它只做逐元素乘法/广播，不会沿 K 做矩阵乘归约。
### tl.sum(x, axis)
```python
block_sum = tl.sum(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 归约结果
- **约束**: 被归约轴必须是当前 block tensor 的静态维度；越界 lane 应在 load 时用 `other=0.0` 清零。
### tl.max(x, axis)
```python
max_val = tl.max(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 最大值
- **约束**: 越界或无效 lane 应在 load 时使用 `other=-float("inf")`，不要先越界 load 再用 `tl.where` 修正。

### tl.min(x, axis)
```python
min_val = tl.min(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 最小值
- **约束**: 越界或无效 lane 应在 load 时使用 `other=float("inf")`，不要先越界 load 再用 `tl.where` 修正。
### tl.maximum(x, y)
```python
z = tl.maximum(x, y)
```
- **参数**: `x`, `y` - 输入张量或标量
- **返回**: 逐元素最大值
- **用途**: elementwise clamp、ReLU、max epilogue 等。
### tl.minimum(x, y)
```python
z = tl.minimum(x, y)
```
- **参数**: `x`, `y` - 输入张量或标量
- **返回**: 逐元素最小值
- **用途**: elementwise clamp、min epilogue 等。
### tl.where(condition, x, y)
```python
result = tl.where(mask, data, 0.0)
```
- **参数**:
  - `condition`: 条件张量
  - `x`, `y`: 选择值
- **返回**: 根据条件选择的值
- **用途**: SIMD 友好的条件选择
- **Ascend 约束**:
  - `tl.where` 适合选择数值，不要依赖它构造可能越界的 pointer 后再 load/store。
  - 对内存访问边界，应优先把条件合进 `tl.load` / `tl.store` 的 `mask`。
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

## 7. 编译期与循环控制 API

### tl.constexpr
```python
BLOCK_SIZE: tl.constexpr = 1024
```
- **用途**: 标记编译时常量参数
- **约束**:
  - 必须在 kernel 函数签名中声明。
  - 从 shape 派生的 `K_TOTAL`、`WINDOW`、`BLOCK_POS` 等静态大小应在 host wrapper 中算好，再作为 `tl.constexpr` meta 参数传入。
  - 不要在 `@triton.jit` 内用 runtime 参数构造 `tl.zeros/full/reshape` 的 shape 或 `tl.static_range` 的上界。
### tl.static_range(start, end, step=1)
```python
for k0 in tl.static_range(0, K_TOTAL_PADDED, BLOCK_K):
    ...
```
- **用途**: 编译期展开固定小循环或固定 tile 循环。
- **约束**:
  - `start`、`end`、`step` 必须是编译期可确定值。
  - 对动态真实边界，使用静态 padded 上界循环，并在循环体内用 mask 过滤无效 lane。
  - 展开次数建议 `<=32`，`32~64` 需要谨慎；可能超过 `64` 时改用 `tl.range`、`tl.dot` K tile 或两阶段 reduction。
  - large reduction 不要退化成一输出一 program 的长串行循环；优先让一个 program 覆盖多个输出位置/列，或拆成两阶段 reduction。
### tl.range(start, end, step=1)
```python
for k0 in tl.range(0, K_TOTAL, BLOCK_K):
    ...
```
- **用途**: 在 kernel 内表达 runtime 边界的循环，常用于动态 K 轴、分块 reduction 或 grid-stride 处理。
- **约束**:
  - 循环体内仍要使用 mask 处理尾块和越界 lane。
  - 不要在需要 runtime 边界的场景使用 Python `range` 或长 `tl.static_range` 展开。
  - 若循环边界其实是很小的编译期常量，才考虑 `tl.static_range`。
## 8. 块分配优化 API

### tl.swizzle2d(i, j, size_i, size_j, group_size)
```python
task_i, task_j = tl.swizzle2d(block_i, block_j, NUM_BLOCKS_I, NUM_BLOCKS_J, GROUP_SIZE)
```
- **参数**:
  - `i`, `j`: 原始块索引
  - `size_i`, `size_j`: 总块数
  - `group_size`: 分组大小(通常为2/4/8)
- **返回**: 重排后的块索引 (task_i, task_j)
- **用途**: 2D块重排,提升缓存局部性
- **适用场景**: 矩阵乘法等多维块计算,改善数据复用
- **注意**: 仅支持行优先(i方向)分组,列优先需手动实现
