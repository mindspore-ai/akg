---
name: pypto-basics
description: "PyPTO 编程原则与核心模式"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "all"
---

# PyPTO 编程原则

## 原则 0：先抽取 Torch 基线的语义合同

写 PyPTO 前，先把 baseline `forward` 变成“可执行语义合同”，再做实现与优化。

最少做 5 件事：
1. 写出数学式（谁参与计算、谁是被规约对象、输出定义是什么）。
2. 核对 API 语义而不是变量名（`input/target/prediction` 这些名字经常误导）。
3. 固化规约合同（`sum`、`batchmean`、或 `mean` 语义=`sum/count`，并明确规约轴与 `keepdim`）。
4. 判断是否非对称（`A||B` 与 `B||A` 是否等价）。
5. 做 1 组语义自检样例（优先用非对称输入，先验方向是否正确）。

注意：性能规则（tile/loop）只能在语义合同确定后应用，不能反向决定语义。

## 原则 1：静态 shape

kernel 一切在编译时确定。**工厂函数**封装 kernel，shape 和标量参数（eps、slope 等）作为闭包传入。forward 直接传 torch.Tensor。**ModelNew.__init__ 签名必须与原始 Model 一致**，shape 在 forward 中获取。

补充（高优先级）：
- benchmark 的 `get_inputs/get_init_inputs` 在单次任务里是固定参数；把它当静态合同，不要过度通用化。
- 先做“固定参数抽取”：从题目文件读取本次 `get_init_inputs` 的返回值，作为本次任务常量（例如 `dim=1`）。
- 题目里的注释如 `Example, change to desired ...` 视为数据集说明噪声，不是本次实现要求。
- 对固定 `dim` 的任务，生成**单一固定 dim kernel**；不要在一个 kernel 里写 `if dim == 0/1/2` 分支。
- 若确实要支持多个 dim，使用多个工厂函数/多个 kernel 分别生成，不要把多语义揉进同一 kernel。

## 原则 2：forward 的职责

1. **Assert**：`assert x.dim() == N`、`assert tuple(x.shape) == (...)`
2. **Reshape**（如需）：torch reshape 为 kernel 能处理的 shape
3. **调用 kernel**：传入 contiguous 张量，reshape 回原始 shape 返回

forward 内禁止 torch 计算。kernel 内不能 reshape。
输出语义（`keepdim`/是否 squeeze）要与 baseline 直接对齐；不要先改语义再靠额外 `squeeze/unsqueeze` 回补。

## 原则 3：选择维度

| 算子类型 | forward 策略 | kernel 维度 |
|----------|-------------|-------------|
| Elementwise / 简单 Loss | `reshape(-1)` | 1D |
| GroupNorm / InstanceNorm | `reshape(flat_batch, hidden)` | 2D |
| BatchNorm / RMSNorm | `reshape(B, C, -1)` | 3D |
| Batched matmul（同 batch） | 保持 3D，不需要 loop | 3D |
| 2D Matmul | 保持 2D | 2D |
| 单轴归约 | 保持原始维度 | 原维 |

补充：
- Elementwise 算子无数据依赖，**优先 flatten 为 1D**；loop 仅由 `auto_tiles > 2048` 触发，与维度无关。
- 保持高维的唯一理由是业务语义需要（如后续算子依赖布局），而非 tile 约束。

补充：
- 多轴规约若轴连续且中间结果不被其他算子使用，优先在 forward 合并为单轴后再规约（例如 `H,W -> HW`）。

## 原则 4：tile 双约束

1. `prod(tile_shape)` ≤ 16384
2. `auto_tiles = prod(ceil(shape[i]/tile[i]))` ≤ 2048

tile 参数个数 = 被操作 tensor 的 rank。常用：
- 1D: `(8192)` | 2D: `(1, 16384)` | 3D: `(1, 1, 16384)` 或 `(1, 16, 256)`（单轴 3D 归约常见起步）
- matmul: `set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)`
- **核心红线**：Skill/示例中的 shape、tile、BLOCK 常量（如 16384/8192/4096）**只能作参考**。必须按当前任务的输入维度重新计算，**禁止直接照抄**。
- **经验规则**：优先避免明显 `tile[i] > shape[i]` 的参数浪费；如确需使用，必须有明确理由（例如实测收益）。

## 原则 5：运算符规则

`+` `*`：标量任意位置。**`-` `/`：tensor 必须在左。** `1.0 - x` crash → 用 `x * (-1.0) + 1.0`。函数调用第一参数必须 Tensor。

## 原则 6：loop

能不用 loop 就不用。loop 场景：auto_tiles > 2048（优先原因）/ 极大归约轴 / matmul M 轴（按对数刻度先定中段 `loop_count`，常见先试 `16/32`，再反推 `BASIC_BATCH=ceil_div(m, loop_count)`；必要时再扩到 `8/64`）。**不嵌套，沿最外轴。** view shape 只能用编译期常量。

---

# 核心编程模式

**模块名 `pypto`**（不是 `pyto`）。

```python
import os, pypto, torch
_PYPTO_RUN_MODE = int(os.getenv("AIKG_PYPTO_RUN_MODE", "0"))
_PYPTO_RUNTIME_DEBUG_MODE = int(os.getenv("AIKG_PYPTO_RUNTIME_DEBUG_MODE", "0"))
```

## 模式 A：Elementwise

**关键判断：`auto_tiles = prod(ceil(shape[i]/tile[i]))` 必须 ≤ 2048。** 超过则必须 loop。

**优先策略**：Elementwise 无数据依赖，**优先 flatten 为 1D**；loop 仅由 `auto_tiles > 2048` 触发，与维度无关。

**小矩阵**（flatten 后 auto_tiles ≤ 2048）：`reshape(-1)` → 1D kernel。
**大矩阵**（如 16384×4096，flatten 后 auto_tiles > 2048）：**1D + loop**（或 2D + loop 仅当语义需要保持 2D）。

```python
# 小矩阵：ELU（16×16384=262144, tile 8192, auto_tiles=32）
def create_elu_kernel(flat_size, alpha):
    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((flat_size,), pypto.DT_FP32)) -> ...:
        output = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        pos = pypto.maximum(x, 0.0)
        neg = pypto.minimum(x, 0.0)
        output[:] = pos + (pypto.exp(neg) - 1.0) * alpha
        return output
    return kernel

# 大矩阵 1D 版本：scalar mul（16384×4096，flatten 后 loop）
def create_scalar_mul_kernel_1d(m, n, s):
    flat_size = m * n
    TARGET_LOOP_COUNT = 32
    BASIC_BATCH = ceil_div(flat_size, TARGET_LOOP_COUNT)
    num_iters = ceil_div(flat_size, BASIC_BATCH)
    @pypto.frontend.jit(...)
    def kernel(a: pypto.Tensor((m, n), pypto.DT_FP32)) -> ...:
        a_flat = pypto.view(a, [flat_size], [0])
        c = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        for bi in pypto.loop(0, num_iters, 1, name="LOOP", idx_name="bi"):
            off = bi * BASIC_BATCH
            chunk = pypto.view(a_flat, [BASIC_BATCH], [off])
            pypto.assemble(chunk * s, [off], c)
        return pypto.view(c, [m, n], [0])
    return kernel

# 大矩阵 2D 版本：仅当语义需要保持 2D 布局时使用
def create_scalar_mul_kernel_2d(m, n, s):
    TARGET_LOOP_COUNT = 32
    BASIC_BATCH = ceil_div(m, TARGET_LOOP_COUNT)
    num_iters = ceil_div(m, BASIC_BATCH)
    @pypto.frontend.jit(...)
    def kernel(a: pypto.Tensor((m, n), pypto.DT_FP32)) -> ...:
        c = pypto.tensor([m, n], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8, 4096)
        for bi in pypto.loop(0, num_iters, 1, name="LOOP", idx_name="bi"):
            off = bi * BASIC_BATCH
            chunk = pypto.view(a, [BASIC_BATCH, n], [off, 0])
            pypto.assemble(chunk * s, [off, 0], c)
        return c
    return kernel
```

## 模式 B：Matmul

- 2D: 沿 M 维 loop（按对数刻度先试中段 `loop_count`，`1~128` 常见从 `16/32` 起步，再反推 `BASIC_BATCH`），`view → matmul → assemble`
- 转置 B：`pypto.matmul(a_chunk, b, pypto.DT_FP32, b_trans=True)`
- 3D batched matmul：`c[:] = pypto.matmul(a, b, pypto.DT_FP32)`，不需要 loop
- 三角/对称/对角矩阵 = 标准 matmul

## 模式 C：Norm + Loop

**GroupNorm / InstanceNorm**（2D）：reshape `(flat_batch, hidden_size)`，tile `(1, 16384)`，loop 沿 flat_batch。`var = sq_sum * inv_hidden - mean * mean`。

**RMSNorm / BatchNorm**（3D）：`(B, C, S)`，tile `(1, 1, 16384)` 或 `(1, 16, 256)`。
- RMSNorm: loop 沿 batch，`sum(dim=1)` 归一化沿 C 轴
- BatchNorm: loop 沿 channel，`sum(dim=0)` + `sum(dim=2)` 跨 batch 和 spatial
- 应用“连续搬运达标后找甜点”规则：RMSNorm 若沿 `C` 轴归约且 `C` 中等（如 64），先保证连续搬运达标，再在 `C` 轴候选里比较（常试 `16/32/64`）。

## 模式 D：Loss → 标量

简单 loss（MSE/Hinge/KLDiv）flatten 到 1D。per-sample loss（Triplet/Cosine）保持 2D，两段 tile。
- 对“方向敏感/非对称”的 loss（散度类、某些概率距离类）：先对齐语义合同，再写公式。
- 例（仅示例）：`F.kl_div(torch.log(pred), target, reduction='batchmean')` 应实现为
  `KL(target || pred)`，并按 `batchmean` 规约。

## 模式 E：大张量全局归约

`pypto.zeros` 累加器 + loop 分块 + `acc[:] = acc + part`。

```python
def ceil_div(a, b):
    return (a + b - 1) // b
```
