---
name: tilelang-ascend-api
description: "TileLang Ascend API 参考手册。提供架构概览、内存分配、数据搬运、矩阵计算、归约、元素级运算、Tile 扩展原语、同步、调度原语等 API 的完整参考。编码范式和优化策略见对应算子类型的 guide。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang Ascend API 最佳实践

## API 速查表

### Kernel 定义

| API | 说明 |
|-----|------|
| `@T.prim_func` | 定义 kernel 函数 |
| `T.Tensor((M, N), dtype)` | 声明张量参数 |
| `T.Kernel(block_num, is_npu=True) as (cid, vid)` | Kernel 启动上下文 |
| `@jit(out_idx=[-1], pass_configs={...})` | JIT 编译装饰器 |
| `T.symbolic('K', 'int32')` | 动态 shape |

### 内存分配

| API | 说明 | 模式 |
|-----|------|------|
| `T.alloc_shared(shape, dtype)` | shared 层级（编译器自动判断 L1/UB） | Developer |
| `T.alloc_fragment(shape, dtype)` | fragment 层级（编译器自动判断 L0A/B/C） | Developer |
| `T.alloc_var(dtype, init=...)` | 标量变量 | Developer |
| `T.alloc_ub / T.alloc_L1 / T.alloc_L0A/L0B/L0C` | 显式指定存储层级 | Expert |

### 数据搬运与计算

| API | 说明 |
|-----|------|
| `T.copy(src, dst)` | GM/L1/UB/L0 之间搬运数据 |
| `T.tile.atomic_add(dst_gm, src_local)` | 将本地 tensor 原子累加到 GM；V1 支持 local/UB → GM |
| `T.gemm_v0(A, B, C, transpose_A, transpose_B, init)` | 标准 GEMM |
| `T.mma(A, B, C, init)` | NPU MMA 指令 |
| `T.reduce_sum/max/min(buffer, out, dim)` | 按维度归约 |

### 循环与调度

| API | 说明 |
|-----|------|
| `T.serial(N)` / `T.unroll(N)` | 普通循环 / 循环展开 |
| `T.Parallel(ext0, ext1, ...)` | 元素级并行循环 |
| `T.Pipelined(range, num_stages=N)` | 流水线并行 |
| `T.Persistent(domain, wave_size, index)` | 持久化调度 |

### 同步

| API | 说明 |
|-----|------|
| `T.set_flag / T.wait_flag` | 核内流水线同步 |
| `T.barrier_all() / T.pipe_barrier(pipe)` | 管线屏障 |
| `T.set_cross_flag / T.wait_cross_flag` | 核间同步 |
| `T.sync_all()` | 全局同步 |

### 常用 pass_configs

| 配置项 | 说明 |
|-------|------|
| `TL_ASCEND_AUTO_SYNC: True` | 自动同步插入 |
| `TL_ASCEND_MEMORY_PLANNING: True` | 自动内存规划 |
| `TL_ASCEND_AUTO_CV_COMBINE: True` | 自动 CV 分离（核间流水线） |
| `tl.ascend_auto_cross_core_sync: True` | 自动核间同步（核间流水线） |

---

## 计算原语：GEMM、归约与 Tile 扩展操作

---

### 1. 矩阵计算（GEMM）

#### T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)

块级矩阵乘操作，计算 C += op(A) × op(B)。A、B 位于 shared 层级，C 位于 fragment 层级。

**参数**：

- `A`：左输入矩阵（shared 层级）
- `B`：右输入矩阵（shared 层级）
- `C`：结果累加输出矩阵（fragment 层级）
- `transpose_A`：是否转置 A（默认 False）
- `transpose_B`：是否转置 B（默认 False）
- `init`：是否在计算前将 C 清零（默认 False）。第一次迭代需要清零，后续累加。

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

for k in T.serial(loop_k):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)
    T.barrier_all()
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
    T.barrier_all()
T.copy(C_L0, C[bx * block_M, by * block_N])
```

**带转置的用法**：

```python
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
```

#### T.mma(A, B, C, init=False)

NPU 级别的矩阵乘累加指令，比 `gemm_v0` 更底层。不支持 `transpose_A`/`transpose_B`。通常配合 `T.alloc_L0A`/`T.alloc_L0B` 和 `T.annotate_layout` 使用。

```python
A_L0 = T.alloc_L0A([block_M, block_K], dtype)
B_L0 = T.alloc_L0B([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
T.annotate_layout({A_L1: make_zn_layout(A_L1), B_L1: make_zn_layout(B_L1)})
T.mma(A_L0, B_L0, C_L0, init=True)
```

---

### 2. 归约操作

#### T.reduce_sum(buffer, out, dim=-1, clear=True, real_shape=None)

#### T.reduce_max(buffer, out, dim=-1, clear=True, real_shape=None)

#### T.reduce_min(buffer, out, dim=-1, clear=True, real_shape=None)

Ascend fast-path reduce 原语，主要服务于 UB tile / slice buffer 场景。

**参数**：

- `buffer`：输入 buffer 或 buffer slice
- `out`：目的输出 buffer 或 buffer slice
- `dim`：reduce 轴
- `clear`：是否在计算前初始化输出
- `real_shape`：2D slice buffer 的逻辑有效范围；未设置时默认使用物理 buffer 形状

**当前支持范围**：

- 1D buffer：`0 / -1`
- 2D buffer：`0 / 1 / -1 / -2`
- 3D buffer：仅支持 trailing-tile 轴 `0 / 1 / -1 / -2`

**`clear` 语义**：

- `clear=True`：先初始化输出，再写入 reduce 结果
- `clear=False`：将 reduce 结果 merge 到已有输出
  - `reduce_sum`：`new_out = old_out + reduced_result`
  - `reduce_max`：`new_out = max(old_out, reduced_result)`
  - `reduce_min`：`new_out = min(old_out, reduced_result)`

**输出 shape 约束**（以 2D 输入 `[M, N]` 为例）：

- `dim=-1`：输出可为 `[M]` 或 `[M, 1]`
- `dim=0`：输出可为 `[N]` 或 `[1, N]`
- 对设置了 `real_shape` 的 2D slice buffer，当前前端还兼容部分 physical-layout 输出形式，例如 `[physical_cols]` 或 `[1, physical_cols]`

**使用建议**：

- `clear` 和 `real_shape` 同时支持关键字传参和兼容的 positional 传参形式
- 推荐优先使用关键字形式，以获得更清晰的可读性
- 非法 `dim`、非法 `real_shape`、非法输出 shape 会在前端直接报错，而不是静默进入后端

**典型用法**：

```python
# Softmax / attention 场景
T.reduce_max(acc_s_ub, m_i, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)

# clear=False merge 语义
T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1, clear=False)

# slice buffer + real_shape
T.reduce_max(in_shared, out_shared, dim=-1, real_shape=[4, 4])
```

---

### 3. Element-wise 运算（Developer 模式 T.Parallel）

在 `T.Parallel` 循环内使用符号 API，跨平台兼容。

```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

**浮点单目运算**：

| 运算 | 算符表达 |
|------|---------|
| 绝对值 | `T.abs(x)` |
| 指数 | `T.exp(x)` |
| 对数 | `T.log(x)` |
| 开平方 | `T.sqrt(x)` |
| 平方根倒数 | `T.rsqrt(x)` |
| ReLU | `T.max(a, 0)` |

**浮点双目运算**：`+`, `-`, `*`, `/`, `T.min(a, b)`, `T.max(a, b)`

**整形运算**：`~`(位非), `<<`, `>>`, `&`(位与), `|`(位或)

**向量-标量运算与广播**：

```python
# 向量-标量
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + 1

# 行广播
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i]  # b_ub.shape = (block_M // VEC_NUM,)

# 维度不匹配广播
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = b_ub[j] + 5  # b_ub 是 1D，c_ub 是 2D
```

**列切分模式**：

```python
for i in range(block_M // VEC_NUM):  # 行顺序
    for j in T.Parallel(block_N):    # 列并行
        c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

---

### 4. Tile 扩展原语（Expert / 混合模式 T.tile.xxx）

`T.tile.xxx` 系列接口直接触发 Tile 级的 Ascend 操作。它们既可用于全手动 Expert 模式，也可在 Developer pass_configs 下作为混合模式原语使用。

#### 4.1 基础算术

| API | 功能 | src1 类型 |
|-----|------|----------|
| `T.tile.add(dst, src0, src1)` | dst = src0 + src1 | buffer 或 scalar |
| `T.tile.sub(dst, src0, src1)` | dst = src0 - src1 | buffer 或 scalar |
| `T.tile.mul(dst, src0, src1)` | dst = src0 * src1 | buffer 或 scalar |
| `T.tile.div(dst, src0, src1)` | dst = src0 / src1 | buffer 或 scalar |
| `T.tile.max(dst, src0, src1)` | dst = max(src0, src1) | buffer 或 scalar |
| `T.tile.min(dst, src0, src1)` | dst = min(src0, src1) | buffer 或 scalar |

#### 4.2 单目运算

| API | 功能 |
|-----|------|
| `T.tile.exp(dst, src0)` | dst = exp(src0) |
| `T.tile.ln(dst, src0)` | dst = ln(src0) |
| `T.tile.abs(dst, src0)` | dst = abs(src0) |
| `T.tile.reciprocal(dst, src0)` | dst = 1/src0 |
| `T.tile.sqrt(dst, src0)` | dst = √src0 |
| `T.tile.rsqrt(dst, src0)` | dst = 1/√src0 |
| `T.tile.relu(dst, src0)` | dst = max(0, src0) |

#### 4.3 需要额外参数的运算

| API | 功能 |
|-----|------|
| `T.tile.leaky_relu(dst, src0, scalar)` | Leaky ReLU，scalar 为负斜率系数 |
| `T.tile.axpy(dst, src0, scalar)` | dst = scalar * src0 + dst |
| `T.tile.sin(dst, src0)` | dst = sin(src0) |
| `T.tile.cos(dst, src0)` | dst = cos(src0) |

#### 4.4 复合运算

| API | 功能 |
|-----|------|
| `T.tile.mul_add_dst(dst, src0, src1)` | dst = src0 * src1 + dst（融合乘加） |
| `T.tile.silu(dst, src0)` | dst = src0 * sigmoid(src0)（SiLU/Swish 激活） |

**说明**：
- `mul_add_dst` 执行融合乘加操作，将 src0 和 src1 相乘后加到 dst 上
- dst 既作为输入（累加器）也作为输出
- 支持 half、float 类型（Atlas A2/A3）
- 也支持 int16_t、uint16_t、int32_t、uint32_t（Atlas 200I/500 A2）

- `silu` 执行 SiLU (Swish) 激活函数: x * sigmoid(x)
- 支持 half、float 类型（Atlas A2/A3）

#### 4.5 逻辑运算

| API | 功能 |
|-----|------|
| `T.tile.bitwise_and(dst, src0, src1)` | dst = src0 & src1 |
| `T.tile.bitwise_or(dst, src0, src1)` | dst = src0 \| src1 |
| `T.tile.bitwise_not(dst, src0)` | dst = ~src0 |
| `T.tile.bitwise_xor(dst, src0, src1)` | dst = src0 ^ src1 |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | 左移操作 |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | 右移操作 |


#### 4.6 比较操作

###### T.tile.compare(dst, src0, src1, mode)

逐元素比较，结果为 bit mask（1=true，0=false）。src1 可以是 buffer 或 scalar。

**mode 取值**：`"EQ"`, `"NE"`, `"GT"`, `"GE"`, `"LT"`, `"LE"`

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")   # tensor vs tensor
T.tile.compare(c_ub, a_ub, 1.0, "GT")     # tensor vs scalar
```

#### 4.7 选择操作

###### T.tile.select(dst, selMask, src0, src1, selMode)

根据 selMask 的比特位选取元素。bit=1 选 src0，bit=0 选 src1。

**selMode 取值**：

- `"VSEL_CMPMASK_SPR"`：根据 compare mask 选择
- `"VSEL_TENSOR_SCALAR_MODE"`：tensor 和 scalar 之间选择
- `"VSEL_TENSOR_TENSOR_MODE"`：两个 tensor 之间选择

```python
T.tile.select(c_ub, selmask_ub, a_ub, b_ub, "VSEL_CMPMASK_SPR")
T.tile.select(c_ub, selmask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
```

#### 4.8 gather_mask

###### T.tile.gather_mask(dst, src, src1Pattern)

根据 mask 模式收集元素。

**固定模式**（src1Pattern 为字符串）：

- `"P0101"`：按偶数索引  `"P1010"`：按奇数索引
- `"P0001"/"P0010"/"P0100"/"P1000"`：每四个取一个
- `"P1111"`：取全部

**自定义模式**（src1Pattern 为 buffer）：按索引选取。

```python
T.tile.gather_mask(b_ub, a_ub, "P0101")
```

#### 4.9 精度转换

###### T.tile.cast(dst, src, mode, count)

**mode 取值**：`"CAST_NONE"`, `"CAST_RINT"`, `"CAST_FLOOR"`, `"CAST_CEIL"`, `"CAST_ROUND"`, `"CAST_TRUNC"`, `"CAST_ODD"`

```python
T.tile.cast(b_ub, a_ub, "CAST_RINT", 4096)
```

#### 4.10 数据操作

| API | 功能 |
|-----|------|
| `T.tile.fill(buffer, value)` | 用 value 填充 buffer |
| `T.tile.createvecindex(dst, first_value)` | 创建从 first_value 开始的向量索引序列 |
| `T.tile.transpose(dst, src)` | 16×16 二维矩阵数据块转置 |
| `T.tile.gather(dst, src, src_offset, src_base_addr)` | 按偏移收集数据 |
| `T.tile.arith_progression(buffer, first_value, diff_value, count)` | 生成等差数列 |

#### 4.10 原子操作

###### T.tile.atomic_add(dst, src)

将本地 tensor tile 原子累加到 GM 目标区域。该 API 是 Ascend 专属的 `T.tile` 原语，不等价于主仓 GPU 风格的全局 `T.atomic_add`。

**V1 支持范围**：

- `dst` 必须是 GM/global buffer、buffer load 或 region
- `src` 必须是本地 tensor，当前主要面向 UB/shared buffer 和 L0C/fragment buffer
- `src` 与 `dst` dtype 必须一致
- 支持 1D 和 2D tile region 的 local -> GM 原子累加
- 不支持 `return_prev`、`memory_order`、`use_tma`、常量 src 或任意表达式 src

**支持的数据类型**：

int8, int16, float16, bfloat16, int32, float32

**使用建议**：

- 如果业务语义是从 0 开始累加，调用 kernel 前或 kernel 内需要显式清零 GM 输出。
- 在混合模式下可配合自动同步和内存规划使用，不要求手写 `T.Scope("V")` 或 `T.barrier_all()`。

**UB -> GM 示例**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

src_ub = T.alloc_ub((tile_n,), "float32")
T.tile.fill(src_ub, 1.0)
T.tile.atomic_add(C[0], src_ub)
```
示例中的pass_config只是最小用法。在混合模式或需要自动 C/V 分离时，可以同时开启 `TL_ASCEND_AUTO_CV_COMBINE`；如果存在 C/V 核间依赖，再配合 `TL_ASCEND_AUTO_CV_SYNC`。

**L0C -> GM 示例**：

适用于矩阵计算结果需要原子累加到 GM 的场景，如多 block/core 的 GEMM 累加。

```python
src_l0c = T.alloc_L0C((block_M, block_N), dtype)
T.gemm_v0(..., ..., src_l0c, init=True)
T.tile.atomic_add(C[..., ...], src_l0c)
```

**底层实现**：

底层会生成 Ascend C 的 DMA atomic add 语义：开启 `SetAtomicAdd<T>()`，执行 local -> GM 的 `DataCopyPad`，再通过兼容 helper 关闭 atomic 状态。
#### 4.11 排序操作

###### T.tile.sort(dst, src, actual_num)

**参数**：

  - dst：存储排序后结果的目标缓冲区(val0, index0, val1, index1 ,...)
  - src：源操作数，待排序数据(val0, val1, val2, ...)
  - actual_num：src 中实际参与排序的元素数量

**功能**：排序函数，将任意长度数据按照数值大小进行一次性降序排序

**举例**：

```
# 对131个数进行排序
# 131向上对齐到160，src.shape = (1, 160), actual_num = 131
T.tile.sort(dst, src, actual_num)
```

**注意事项**：
  - `dst`与 `src` 数据类型相同，仅支持float32和float16数据类型
  - `src` 的大小需要满足32或32的整数倍

###### T.tile.merge_sort(dst, src0, src1, src2=None, src3=None)

将多个已排序数据块合并，支持 2/3/4-way 归并。输入/输出均为 value-index pair 格式。

```python
T.tile.merge_sort(merge_dst, src0, src1)            # 2-way
T.tile.merge_sort(merge_dst, src0, src1, src2)       # 3-way
T.tile.merge_sort(merge_dst, src0, src1, src2, src3) # 4-way
```

###### T.tile.topk(dst, src, K, actual_num)

**参数**：

  - dst：存储TopK结果的目标缓冲区(val0, index0, val1, index1 ,...)
  - src：包含输入数据的源缓冲区(val0, val1, val2, ...)
  - K：前K个排序结果
  - actual_num：实际参与排序的元素个数

**功能**：执行 TopK 操作，实现对源数据的一次性从大到小排序，选择前K个元素，以（数、索引）的方式输出

**举例**:

```
# 对41个数进行排序，选择前10个数
# 需要使41向上对齐至32 * 2 = 64，K = 10, actual_num = 41
# topk_global.shape = (1, 20)sort_result.shape = (1, 64)
T.tile.topk(topk_global, sort_result, K, actual_num)
```

**注意事项**：
  - `src` 的大小需要满足32或32的整数倍

#### 4.12 两种编程范式对比

```python
# 方式一：T.Parallel + 符号 API（推荐，跨平台兼容）
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    b_ub[i, j] = T.exp(a_ub[i, j])

# 方式二：T.tile 扩展原语（Expert / 混合模式，直接触发硬件指令）
T.tile.exp(b_ub, a_ub)
```

---

## Kernel 定义、内存分配与数据搬运

---

### 1. Kernel 定义与启动

#### @T.prim_func

定义一个 TileLang kernel 函数。参数类型为 `T.Tensor` 或 `T.Buffer`。

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((M, N), dtype),
    B: T.Tensor((M, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    ...
```

**支持的 dtype**：`float16, float32, bfloat16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`

#### 动态 shape 符号

**T.symbolic(name, dtype)**：创建可直接使用的 tir.Var
  ```python
  K = T.symbolic('K', 'int32')
  @T.prim_func
  def bar(A: T.Tensor((K,), 'float32')):
      for i in T.serial(K):
          ...
  ```


#### T.Kernel

定义 kernel 运行上下文，创建 tile block 与逻辑核的绑定。

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    ...
```

- **cid**：计算任务 ID，范围 [0, block_num)
- **vid**：Vector 单元索引（0 或 1），A2/A3 架构 CV 核配比可为 1:2 或 1:1
- **VEC_NUM**：通常设为 2，表示每个 AI Core 有 2 个 Vector 计算单元

#### @jit 装饰器

触发即时编译，将 kernel 编译为 NPU 可执行代码。

```python
@jit(out_idx=[-1], pass_configs=pass_configs)
def tile_add(M, N, block_M, block_N, dtype='float'):
    @T.prim_func
    def main(...):
        ...
    return main
```

**参数**：
- `out_idx`：指定输出参数索引，如 `[-1]` 表示最后一个参数为输出
- `workspace_idx`：工作空间参数索引（如 Flash Attention 中 `workspace_idx=[4,5,6]`）
- `pass_configs`：编译配置选项

**常用 pass_configs**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离（核间流水线需要）
}
```

---

### 2. 内存分配原语

#### Developer 模式

TileLang 对存储层级进行了抽象，分为 Global、shared 和 fragment 三个级别。在 Ascend 平台中，shared 层级对应 L1 Buffer 和 Unified Buffer，fragment 层级对应 L0A/L0B/L0C Buffer。用户无需指定具体硬件存储，TileLang 编译器会根据程序上下文自动识别。

###### T.alloc_shared(shape, dtype)

分配 shared 层级的存储空间。

```python
A_L1 = T.alloc_shared((block_M, block_K), dtype)
```

###### T.alloc_fragment(shape, dtype)

分配 fragment 层级的存储空间。

```python
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
```

###### T.alloc_var(dtype, init, scope='local.var')

分配标量变量，支持初始化。适用于标志位、计数器、临时标量。

```python
flag = T.alloc_var("bool", init=False)
counter = T.alloc_var("int32", init=1)
b = T.alloc_var("int32", init=a)  # 用另一个变量的值初始化
```

#### Expert 模式

显式指定存储位置，适用于需要精确控制内存分配的场景。

| API | 存储层级 | 说明 |
|-----|---------|------|
| `T.alloc_ub(shape, dtype)` | Unified Buffer (UB) | Vector 计算 |
| `T.alloc_L1(shape, dtype)` | L1 Buffer | 片上缓存 |
| `T.alloc_L0A(shape, dtype)` | L0A Buffer | Cube 左矩阵 |
| `T.alloc_L0B(shape, dtype)` | L0B Buffer | Cube 右矩阵 |
| `T.alloc_L0C(shape, dtype)` | L0C Buffer | Cube 输出/累加 |

**实际使用示例**：

```python
A_L1 = T.alloc_L1([block_M, block_K], dtype)
B_L1 = T.alloc_L1([block_K, block_N], dtype)
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)
```

---

### 3. 数据搬运原语

#### T.copy(src, dst)

在不同内存层级之间搬运 tile 数据块。支持 tir.Buffer、BufferLoad、BufferRegion 类型。

**支持的搬运路径**：

| src | dst | 说明 |
|-----|-----|------|
| GM | L1 | Global Memory → L1 Buffer |
| L1 | L0A | L1 Buffer → L0A Buffer（Cube 左矩阵）|
| L1 | L0B | L1 Buffer → L0B Buffer（Cube 右矩阵）|
| L0C | GM | L0C Buffer → Global Memory |
| GM | UB | Global Memory → Unified Buffer |
| UB | GM | Unified Buffer → Global Memory |
| UB | UB | Unified Buffer → Unified Buffer |
| UB | L1 | Unified Buffer → L1 Buffer |

**使用示例**：

```python
# GM → L1
T.copy(A[bx * block_M, k * block_K], A_L1)

# GM → UB（vid 切分）
T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

# UB → GM
T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

# L0C → GM
T.copy(C_L0, C[bx * block_M, by * block_N])

# BufferRegion 切片搬运
T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
```

---

## 调度、同步

---

### 1. 循环原语

#### T.serial(N) / T.serial(start, end, step)

普通 for 循环。

```python
for i in T.serial(N):        # 0..N-1
for i in T.serial(0, N, 2):  # 0, 2, 4, ...
```

#### T.unroll(N)

针对小循环次数进行循环展开。TileLang 将展开提示传递给 TIR。

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

#### While 循环

循环条件需要是 TIR expression。TileLang 检测出死循环会编译报错。

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

**Break 和 Continue**：在 T.serial/T.unroll/T.Parallel/while 循环中均可使用。

---

### 2. T.Pipelined

实现计算/搬运的流水线并行，通过预取来掩盖内存访问延迟。

#### 语法

```python
for var in T.Pipelined(range, num_stages=N):
    ...
```

- `range`：迭代次数
- `num_stages`：预取阶段数（小于 range-1 的正整数）

#### 核内流水线（Intra-core）

```python
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)

    T.barrier_all()
    if k == 0:
        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
    else:
        T.gemm_v0(A_L1, B_L1, C_L0)

    T.barrier_all()
```

`num_stages=2` 时执行顺序：

| Time | Copy A/B | Compute |
|------|----------|---------|
| t₀ | copy_A_0, copy_B_0 | |
| t₁ | copy_A_1, copy_B_1 | |
| t₂ | copy_A_2, copy_B_2 | gemm_0 |
| t₃ | copy_A_3, copy_B_3 | gemm_1 |
| t₄ | | gemm_2 |
| t₅ | | gemm_3 |

#### 核间流水线（Inter-core）

Cube 和 Vector 核之间的流水并行：

```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    T.tile.fill(acc_s_ub, 0.0)
    T.copy(workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
           acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
    ...
```

**注意**：
- 核间流水线与核内流水线不能同时开启
- 使用核间流水线必须开启：`"tl.ascend_auto_cv_combine": True`, `"tl.ascend_auto_cross_core_sync": True`

---

### 3. T.Persistent

优化数据块在 AI Core 间的调度，使相邻数据块交由同一 AI Core 处理，提高缓存命中率。

```python
for bx, by in T.Persistent(domain, wave_size, index):
    ...
```

**参数**：
- `domain`：迭代空间
- `wave_size`：wave 大小（通常为 core_num）
- `index`：当前核的索引（通常为 cid）

**示例**：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    A_L1 = T.alloc_shared((block_M, K_L1), dtype)
    B_L1 = T.alloc_shared((K_L1, block_N), dtype)
    C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

    for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                                core_num, cid):
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * K_L1], A_L1)
            T.copy(B[k * K_L1, by * block_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.copy(C_L0, C[bx * block_M, by * block_N])
```

---

### 4. 同步原语

#### 核内同步

| API | 说明 |
|-----|------|
| `T.set_flag(src, dst, eventId)` | 设置核内流水线同步标志（producer 完成通知） |
| `T.wait_flag(src, dst, eventId)` | 等待核内流水线同步标志（consumer 阻塞等待） |
| `T.barrier_all()` | 所有管线的全局屏障 |
| `T.pipe_barrier(pipe)` | 特定管线的屏障（如 `"MTE3"`, `"V"`） |
| `T.sync_all()` | 全局同步 |

**管线名称**：`"fix"`, `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`

```python
T.set_flag("mte2", "v", 0)
T.wait_flag("mte2", "v", 0)
```

#### 核间同步

| API | 说明 |
|-----|------|
| `T.set_cross_flag(pipe, flag)` | 设置核间同步标志 |
| `T.wait_cross_flag(flag)` | 等待核间同步标志 |

```python
# Cube 核完成后通知 Vector 核
T.set_cross_flag("MTE3", 0)
T.wait_cross_flag(0)
```

> `set_cross_flag` 源码（`ascend.py:114`）还支持第三个参数 `mode`（默认 2），控制同步范围：0=所有 AIC/AIV 之间，1=同组 AIV 之间，2=同组 AIC 和 AIV 之间。

---

### 5. T.Scope

用于标注代码块的执行域。

```python
with T.Scope("C"):   # Cube 域
    ...
with T.Scope("V"):   # Vector 域
    ...
```