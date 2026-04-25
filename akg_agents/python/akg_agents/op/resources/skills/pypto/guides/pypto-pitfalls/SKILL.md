---
name: pypto-pitfalls
description: "PyPTO 常见首次生成错误及正确写法"
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "all"
---

# PyPTO 常见陷阱

## 1. 运算符规则（最高频错误）

`+` `*`：标量任意位置。`-` `/`：tensor 必须在左。函数调用：第一参数必须 Tensor。

```python
1.0 + x            # OK（__radd__）
1.0 - x            # CRASH（__rsub__ 未实现）
1.0 / x            # CRASH（__rtruediv__ 未实现）
pypto.add(1.0, x)  # CRASH（函数调用标量在前）

# 1 - x 正确写法
x * (-1.0) + 1.0   # 推荐

# 标量之间用 Python 运算
neg_delta = -delta  # OK（delta 是闭包 float）
```

## 2. clamp / min(x, d) 实现

`pypto.clamp` 不可用。`min(x, d)` 用双重取反：`-max(-x, -d)`。`pypto.minimum(x, 0.0)` 可用。

```python
# min(abs_diff, delta) — delta 是闭包 float
neg_abs = pypto.mul(abs_diff, -1.0)
clipped = pypto.mul(pypto.maximum(neg_abs, -delta), -1.0)  # = min(abs_diff, delta)
```

**Huber Loss 完整模式**（必须用 clamp，不能简化）：
```python
diff = predictions - targets
abs_diff = pypto.abs(diff)
neg_abs = pypto.mul(abs_diff, -1.0)
clipped = pypto.mul(pypto.maximum(neg_abs, -delta), -1.0)  # min(|d|, delta)
half_sq = clipped * clipped * 0.5
loss = half_sq + abs_diff - clipped  # 完整 Huber 公式
total = pypto.sum(loss, dim=0, keepdim=True)
output[:] = total / flat_size
```

## 3. 工厂函数

标量参数（eps、slope、margin 等）**必须**作为工厂函数参数通过闭包传入 kernel。

3D+2D matmul 时，forward 展平后**传展平维度 nm=N*M** 给工厂，不要分别传 N、M。

## 4. matmul K > 65535

用逐元素乘法 + `pypto.sum(a * b_broadcast, dim=1)` 替代。forward 中 `B.reshape(1, -1)` 使其可广播。

## 5. 距离度量必须 sqrt

`sum(diff*diff)` 是**平方距离**，不是 L2 距离。TripletMarginLoss 等必须 `pypto.sqrt(sum_sq + eps)`。

## 6. tile rank = tensor rank

`set_vec_tile_shapes` 参数个数必须等于被操作 tensor 的 rank。2D tensor 用 2D tile。

## 6.1 盲抄 tile 常量（尤其 16384）

示例里的 `16384`/`8192` 是经验候选，不是固定答案。必须按当前 shape 和归约维重算。

- 常见误用：输入 `(128, 4096)` 却写 `set_vec_tile_shapes(1, 16384)`。
- 更合理候选：`set_vec_tile_shapes(4, 4096)`（归约轴不浪费，且 batch 并行更高）。

要点：
- 优先避免明显 `tile[i] > shape[i]` 的“预算浪费”。
- 示例代码只能借结构，不能照抄 shape/tile 数字。

## 6.2 把“少分段”误读成“归约轴越大越快”

“连续搬运达阈值后再调归约轴”是二级目标，但不是“归约轴 tile 越大越快”。

- 常见误用：直接写 `tile_hidden = hidden`，追求归约轴一次覆盖。
- 典型后果：UB/OoOSchedule 报错（即使语义正确也无法编译）。

正确做法：
- 先满足 `prod(tile_shape) <= 16384` 与 `auto_tiles <= 2048`。
- 若出现 UB/OoOSchedule 报错，优先降档：`16384 -> 8192 -> 4096`。
- 若 `auto_tiles > 2048`，优先改为 loop 分块，不要硬塞更激进 tile。
- 先让连续搬运达到约 `1KB`（经验阈值），再在达标候选里做归约轴甜点比较（常试 `16/32/64`）。

## 6.3 把“搬运优先”误读成“只要更连续就一直加”

连续搬运是先达标，不是无限放大。达到高效区后，继续把预算给非规约轴通常收益很小，反而会挤占规约轴 tile 预算。

正确做法：
- 用经验阈值起步：`contiguous_bytes(tile) >= 1KB`。
- 达标后，把剩余预算用于归约轴甜点搜索，不做“越大越快”假设。
- 若多个候选都达标，优先按实测选甜点（例如 `(1,16,256)` 优于 `(1,64,256)` 的场景）。
- 不要把 loop 的“中段起步”规则直接迁移到 tile；对已知固定形状应优先采用已验证优选值。

## 6.4 把 `shape` 当成连续搬运长度（高频误判）

连续搬运阈值必须按 **tile 的实际连续段** 计算，不能按输入 `shape` 计算。

错误示例：
- `shape=(16,256,256), dim=1, tile=(1,256,64)` 时，写成“连续维是 256，所以已达 1KB”。

正确计算：
- Vec 常见估算：`contiguous_tile_elems = tile[last_axis]`。
- 上例连续搬运应按 `tile[2]=64` 计算：`64*4=256B`（FP32），未达 1KB。
- 若要先满足 1KB 阈值，应让连续维 tile 至少到 256（FP32），再比较规约分段。

## 6.5 混用 cube/vec 算子却只设一次 tile（编译期高频崩溃）

高频误用：
- 先 `set_cube_tile_shapes(...)` 做 `matmul`，随后直接做 `add/mul/expand_clone`。
- 误以为 cube tile 会自动覆盖 vec 算子。

典型报错：
- `ASSERTION FAILED: vecTile.valid()`
- `op [ADD] tile shape not set`

正确做法：
- 把 kernel 拆成阶段：先 cube 阶段（matmul），再 vec 阶段（elementwise/broadcast）。
- 进入 vec 阶段前显式 `pypto.set_vec_tile_shapes(...)`。
- 线性层 `y = x @ w + b` 推荐：`b` 在 forward 先 `reshape(1, -1)`，kernel 用 `expand_clone` 广播后再 `add`。

## 7. 优先内建函数

`pypto.sigmoid`、`pypto.softmax`、`pypto.abs`、`pypto.exp`、`pypto.log`、`pypto.sqrt` 等有内建的直接用，禁止手动实现。

## 8. 不需要 assert flat_size % tile_size == 0

auto-tile 自动处理余数。只有 loop+view 模式需要确保整除性。

## 9. per-sample loss 两段 tile

2D 输入的 per-sample loss：Phase 1 `(4, 4096)` 算 per-sample → Phase 2 `(128, 1)` 跨 batch 归约。

## 10. 条件分支用 maximum + minimum

`pypto.where` 不可用。**`pypto.minimum` 可用！**

```python
# ELU
output[:] = pypto.maximum(x, 0.0) + (pypto.exp(pypto.minimum(x, 0.0)) - 1.0) * alpha
# LeakyReLU
output[:] = pypto.maximum(x, 0.0) + pypto.minimum(x, 0.0) * slope
```

禁止 `maximum(x, f(x))` 做条件选择（正半轴 f(x)>x 时结果错误）。

## 11. 方差公式

`var = sq_sum * inv_count - mean * mean`（E[x²]-E[x]²）。符号反了会 NaN。

## 12. ModelNew.__init__ 签名

必须与原始 Model 一致。shape 在 forward 中获取，不要加到 `__init__` 参数里。

## 12.1 静态任务写成“万能 kernel”（多 dim 分支）

高频误用：
- 在一个 kernel 里写 `if dim == 0/1/2`，试图一次覆盖所有归约维。
- 为了复用分支，先 `keepdim=True` 再在 forward `squeeze` 回去。

为什么不推荐：
- benchmark 单次任务的 `shape/dim` 来自固定 `get_inputs/get_init_inputs`，本质是静态合同。
- 多分支会引入与题目无关的搜索空间，增加错误几率（tile 也更容易被写成无关候选）。
- `Example, change to desired ...` 这类注释是题库说明，不是当前运行要求；据此扩展多 dim 属于过拟合题面文字。

正确做法：
- 对当前任务只实现固定参数路径（例如 `dim=1`）。
- 输出语义直接对齐 baseline（`keepdim` 是否保留按 baseline 来），不做“先改再补”的绕行。
- 固定 `dim` 时，不要把 `dim` 作为 kernel 运行时参数透传；在 kernel 内直接使用固定常量 `dim=<固定值>`。

## 13. 模块名 `pypto`

不是 `pyto`、`pytorch`、`pto`。所有调用以 `pypto.` 开头。

## 14. 基线语义误读（高频且隐蔽）

最危险的错误不是语法，而是“语义合同”读错：代码能编译、甚至能 PASS，但算的不是同一个东西。

高发误读源：
- 把变量名当语义（如 `input/target/predictions`）。
- 把 API 名字当数学方向（尤其非对称目标）。
- 看到 verify 通过就默认语义正确（宽容差可能掩盖方向错误）。

防错流程（先语义、后实现）：
1. 从 baseline `forward` 写出数学式。
2. 单独查 API 合同：参数含义、是否 log 输入、规约定义。
3. 固化规约语义：`sum`、`batchmean`、或 `mean` 语义=`sum/count`，并明确规约轴与输出形状。
4. 标记是否非对称；非对称时明确“参考项”和“被比较项”。
5. 用 1 组可解释样例做语义自检（优先非对称分布）。

示例（KLDiv，仅示例）：
- `F.kl_div(torch.log(pred), target, reduction='batchmean')`
- 若按模板写成 `pred * (log(pred) - log(target))`，方向就反了。

## 15. 连续规约轴不合并，导致多次归约中间开销

当规约目标本质是“连续多轴联合规约”时，直接链式 `sum(dim=...)` 常会产生中间张量与额外调度。

更优候选：
- forward 先把连续规约轴合并（如 `H,W -> HW`，或 `(B,H) -> (B*H)`）。
- kernel 用单次规约完成主归约，再做最终语义规约（如 `batchmean` 的除 `B`）。

高频误区：
- 把“per-sample 两阶段归约”当默认模板，导致本可合并的连续规约也被拆成两段。

正确判定：
- 只有当中间结果必须被其他算子/输出复用时，才保留两阶段。
- 若中间结果仅用于继续规约，优先合并连续规约轴。
