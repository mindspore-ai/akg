---
name: pypto-api
description: "PyPTO 全部 API 签名与约束速查"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
---

# PyPTO API 速查

## Kernel 装饰器

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": _PYPTO_RUN_MODE},
    debug_options={"runtime_debug_mode": _PYPTO_RUNTIME_DEBUG_MODE},
)
def kernel(x: pypto.Tensor(shape_tuple, dtype)) -> pypto.Tensor(shape_tuple, dtype):
    ...
```

## 张量

| API | 用途 | 示例 |
|-----|------|------|
| `pypto.Tensor(shape, dtype)` | 输入/输出类型标注 | `x: pypto.Tensor((m, k), pypto.DT_FP32)` |
| `pypto.tensor(shape_list, dtype)` | kernel 内创建输出 | `output = pypto.tensor([m, n], pypto.DT_FP32)` |
| `pypto.zeros(shape_list, dtype=)` | 零初始化张量（累加器） | `acc = pypto.zeros([1], dtype=pypto.DT_FP32)` |
| `pypto.full(shape, val, dtype, valid_shape=)` | 常量填充张量 | `ones = pypto.full(s, 1.0, pypto.DT_FP32, valid_shape=s)` |

数据类型：`pypto.DT_FP32`、`pypto.DT_INT32`、`pypto.DT_INT64`（INT64 仅用于输入标注）。

## Tile 配置

| API | 约束 |
|-----|------|
| `pypto.set_vec_tile_shapes(*shapes)` | 参数个数 = 被操作张量 rank |
| `pypto.set_cube_tile_shapes(m, k, n, l1, split_k)` | 固定 5 参数 |

**tile 双约束**：
1. `prod(tile_shape)` ≤ 16384
2. `auto_tiles = prod(每维 ceil(shape[i]/tile[i]))` ≤ 2048（每个 op）

- 必须在任何计算操作之前调用。一个 kernel 内可多次切换 tile。
- 若 `auto_tiles > 2048`，优先改为 `loop + view/assemble` 分块实现。
- **vec tile 推荐**：`(8192)` (1D)、`(1, 16384)` (2D)、`(1, 1, 16384)` (3D)
- **cube tile 推荐**：`set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)`

## 分块

| API | 用途 | 约束 |
|-----|------|------|
| `pypto.loop(start, end, step, name=, idx_name=)` | 编译期循环 | 不嵌套、尽量少用 |
| `pypto.view(tensor, shape, offset)` | **切片提取**（等价 `tensor[a:b, c:d]`） | shape 各维为编译期常量，rank 不变，每维 ≤ 输入对应维 |
| `pypto.assemble(chunk, offset, output)` | 写回子块（等价切片赋值） | 无 |

`pypto.view` **不是 reshape**。它是 `tensor[offset[0]:offset[0]+shape[0], ...]` 的等价 API。不能改变维度数，不能改变维度排布。所有 reshape 必须在 forward 中用 torch 完成。

## 算术运算

**运算符规则**：`+` `*` 支持标量在任意位置；`-` `/` 要求 tensor 在左侧（`1.0 - x` crash）。
**函数调用**：`pypto.add`/`sub`/`mul`/`div` 第一参数必须 Tensor。
一元取反 `-x`：不支持，用 `pypto.mul(x, -1.0)` 或 `x * (-1.0)`。
切片赋值：`output[:] = expr`

## 数学函数

| 函数 | 说明 |
|------|------|
| `pypto.exp(x)` | 指数 |
| `pypto.log(x)` | 对数 |
| `pypto.sqrt(x)` | 平方根 |
| `pypto.abs(x)` | 绝对值 |
| `pypto.sigmoid(x)` | sigmoid |
| `pypto.softmax(x, dim=)` | softmax |
| `pypto.maximum(a, b)` | 逐元素最大，b 可以是标量：`pypto.maximum(x, 0.0)` |
| `pypto.minimum(a, b)` | 逐元素最小，b 可以是标量：`pypto.minimum(x, 0.0)` |

有内建函数时直接调用，禁止手写等价公式。

**禁用 API**：`pypto.where`（有 bug）、`pypto.clamp`（不稳定）。条件逻辑用 `maximum`/`minimum` 组合实现。

## 归约

```python
pypto.sum(x, dim=int, keepdim=bool)
pypto.amax(x, dim=int, keepdim=bool)
pypto.amin(x, dim=int, keepdim=bool)
```

**无 `pypto.mean` API。** `mean` 语义请用 `sum * (1.0 / count)` 实现。
**`dim` 只接受单个 `int`，不支持 `list`。** 多轴归约需连续多次调用。
`dim` 在实际实现中应作为闭包常量参与编译；静态任务不要在同一 kernel 内写多 `dim` 运行时分支。

## 矩阵乘法

```python
pypto.matmul(a, b, out_dtype, a_trans=False, b_trans=False)
```

- 2D：`[M,K] @ [K,N] → [M,N]`
- 3D batched：`[B,M,K] @ [B,K,N] → [B,M,N]`（两边必须同 rank 同 batch，**不支持广播**）
- `a_trans` / `b_trans` 支持转置。
- **限制**：每个输入最后一维 ≤ 65535。超过时用 `sum(a * b_broadcast, dim=)` 替代。
- **matmul 几乎必须配合 loop**（M 轴分块），因 matmul 内部展开复杂，不 loop 容易 timeout。例外：M 很小（≤128）时可不 loop。

## 类型转换与索引

| API | 示例 |
|-----|------|
| `pypto.cast(tensor, dtype)` | `pypto.cast(x, pypto.DT_INT32)` |
| `pypto.unsqueeze(tensor, dim)` | `pypto.unsqueeze(x, 1)` |
| `pypto.gather(tensor, dim, index)` | `pypto.gather(log_probs, dim=1, index=idx)` |
| `pypto.expand_clone(tensor, shape)` | 单轴广播，每次只能扩展一个轴 |
