---
name: triton-ascend-case-vector-mask-i32
description: "比较两侧本身多为 `tl.int32`（offset / attn_arg 等），`arith.cmpi` 产出的是 `i1`；多段结果在 **`i1` 张量上做 `&`/`|`** 时， lowering 会在每条逻辑附近插入 **`extui`/`trunci`** 与 `select` 对齐；**每段比较后立刻 `.to(tl.int32)`**，让整条 mask 在 **`i32` 0/1 上 `&`/`|`**，后端更易连续处理 **`vand.i32`/`vor.i32`**。"
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
---

## 任务特征
- 多段 **`tl.int32` 参与比较**（如 `q_off`/`k_off`、`q_attn_arg`/`k_attn_arg`），得到的 **bool 再用 `&` / `|` 拼成 mask**，最后交给 `tl.where`。

## 原因

让 arith.cmpi 的 `i1` 结果在`tensor<…xi1>` 上反复 `andi`/`ori`，Ascend 后端会在每条逻辑与最后的 `arith.select` 之间额外插入`arith.extui` / `arith.trunci`等，把向量的宽度对齐前后使用的`tl.int32`，显式 `.to(tl.int32)`中间 不再在 `i1` 上对齐宽度，vector 上更易使用指令`vand.i32` / `vor.i32`。

## 参考写法

```python
@triton.jit
def mask(...):
    triu = (q_off[:, None] <= k_off[None, :]).to(tl.int32)
    return (
        (triu & ((q_arg[:, None] == k_arg[None, :]).to(tl.int32)
                 | (k_arg[None, :] == 0).to(tl.int32)))
        | (q_off[:, None] == k_off[None, :]).to(tl.int32))
```

## 注意
- mask 语义只需 **0/1**，`tl.int32` 的数据类型来源于比较操作符两边的数据类型。
