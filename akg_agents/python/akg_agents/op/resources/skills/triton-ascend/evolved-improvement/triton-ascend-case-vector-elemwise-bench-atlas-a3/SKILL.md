---
name: triton-ascend-case-vector-elemwise-bench-atlas-a3
description: "Atlas A3 上 Triton vector 一元/二元 `tl.*` ：fp32/fp16/bf16 三种 dtype 下各算子端到端 time (ms)；并给出\"语义等价、精度对齐\"前提下应替换的 triton API 与推荐写法。例如 fp32 上 `tl.exp2(x)` 比 `tl.exp(x*LN2)` 性能表现得要差，可以选择使用 `tl.exp(x*LN2)`。"
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A3"
---


## 注意事项

1. **`exp` / `exp2`**：A3 上二者性能差距显著（fp32 上 `tl.exp2` 比 `tl.exp` 性能表现很差，数学等价 **`exp2(x) = exp(x * LN2)`**，勿与 **`LOG2E`** 混淆，但是要保证精度一致。

---

# 互替换 API 推荐（Atlas A3）

> **精度告警**：以下替换在单算子单元测试上是"语义等价、精度对齐"的，但在算子融合或者网络中使用时，
> 误差可能会累积放大。
> **若替换某条 API 后精度无法对齐，请保留原写法，不要替换。**

## fp32

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.exp2(x)` | **`tl.exp(x * LN2)`**（`LN2=0.6931471805599453`） | 45% |
| `tl.where(x>0, x, 0)` 实现 relu | `(x + tl.abs(x)) * 0.5` 或 `tl.maximum(x, 0)` | 72% |
| `tl.div_rn(x, y)` | `x / y` 或 `tl.fdiv(x, y)` 或 `x * (1.0/y)` | 17% |
| `x * tl.sigmoid(x)` 实现 silu | `x / (1 + tl.exp(-x))` | 14% |
| `tl.sigmoid(x)` | `tl.exp(x) / (1 + tl.exp(x))` | 7% |
| `x * tl.rsqrt(x)` 或 `1.0/tl.rsqrt(x)` 实现 sqrt | `tl.sqrt(x)` 或 `tl.sqrt_rn(x)` | 5-6% |

## fp16

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.where(x < y, x, y)` 实现 minimum | **`tl.minimum(x, y)`** | **57%** |
| `tl.sigmoid(x)` | `tl.exp(x) / (1 + tl.exp(x))` | 10% |
| `tl.rsqrt(x)` | `1.0 / tl.sqrt(x)` | 11% |
| `tl.log2(x)` | `tl.log(x) * LOG2E`（`LOG2E=1.4426950408889634`） | 8% |
| `tl.abs(x)` | `tl.maximum(x, -x)` | 8% |
| `tl.fdiv(x, y)` | `tl.div_rn(x, y)` | 7% |
| `tl.where(x>0, x, 0)` 或 `(x+\|x\|)*0.5` 实现 relu | `tl.maximum(x, 0)` | 7-8% |


## bf16

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.abs(x)` | **`tl.maximum(x, -x)`**（A3 bf16 上 `tl.abs` 严重劣化） | **103%** |
| `tl.where(x>0, x, 0)` 实现 relu | **`tl.maximum(x, 0)`** | **86%** |
| `tl.where(x<y, x, y)` 实现 minimum | **`tl.minimum(x, y)`** | **78%** |
| `2*tl.sigmoid(2*x)-1` 实现 tanh | `1 - 2/(tl.exp(2*x)+1)` 或 `(exp(x)-exp(-x))/(exp(x)+exp(-x))` | 33% |
| `(x + tl.abs(x)) * 0.5` 实现 relu | `tl.maximum(x, 0)` | 21% |
| `tl.exp2(x)` | `tl.exp(x * LN2)` | 21% |
| `tl.sqrt_rn(x)` | `tl.sqrt(x)` | 11% |
| `tl.log2(x)*LN2` 实现 log | `tl.log(x)` | 6% |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 6% |


## 跨 dtype 的二维/特殊场景

| 原写法 | 推荐替换 | 备注 |
|---|---|---|
| `acc / l[:, None]`（`acc: (M, D)`, `l: (M,)`）—— fp32 | `l_recip = 1.0/l; acc * l_recip[:, None]` 或 `acc * (1.0/l)[:, None]` | A3 fp32 上直接除慢 **42%**，把 `M*D` 次 vdiv 降到 `M` 次，剩余转 vmul |
| `acc * (1.0 / l)[:, None]`（fp16） | `l_recip = 1.0/l; acc * l_recip[:, None]` | 内联 `1.0/l` 的写法 fp16 上慢 42%；显式拆出 `l_recip` 更稳|
