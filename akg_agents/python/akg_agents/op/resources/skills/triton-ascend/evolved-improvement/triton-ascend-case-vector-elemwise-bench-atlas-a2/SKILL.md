---
name: triton-ascend-case-vector-elemwise-bench-atlas-a2
description: "Atlas A2 上 Triton vector 一元/二元 `tl.*`：fp32/fp16/bf16 三种 dtype 下各算子端到端 time (ms)；并给出\"语义等价、精度对齐\"前提下应替换的 triton API 与推荐写法。"
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2"
---

# 互替换 API 推荐（Atlas A2）

> **精度告警**：以下替换在单算子单元测试上是"语义等价、精度对齐"的，但在算子融合或者网络中使用时，
> 误差可能会累积放大。
> **若替换某条 API 后精度无法对齐，请保留原写法，不要替换。**

## fp32

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `x * tl.sigmoid(x)` 实现 silu | **`x / (1 + tl.exp(-x))`** | **13%** |
| `tl.sqrt(x)` | `tl.sqrt_rn(x)` / `x*tl.rsqrt(x)` / `1.0/tl.rsqrt(x)` | 5% |
| `2*tl.sigmoid(2x)-1` vs `tl.math.tanh(x)` | **`tl.tanh(x)`** | 4% |

## fp16

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.rsqrt(x)` | **`1.0 / tl.sqrt(x)`** | **24%** |
| `tl.sqrt(x)` | **`tl.sqrt_rn(x)`** 或 `x*tl.rsqrt(x)` / `1.0/tl.rsqrt(x)` | **27%** |
| `tl.exp2(x)` | **`tl.exp(x * LN2)`** | **24%** |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 23% |
| `tl.maximum(x, 0)` 实现 relu | **`(x + tl.abs(x)) * 0.5`** 或 `tl.where(x>0, x, 0)` | **19%-21%** |
| `(exp(x)-exp(-x))/(exp(x)+exp(-x))` 实现 tanh | **`tl.tanh(x)`** 或 `1-2/(exp(2x)+1)` / `2*sigmoid(2x)-1` | **47%** |
| `tl.sigmoid(x)` | `exp(x)/(1+exp(x))` 或 `1/(1+tl.exp(-x))` 或 `0.5*(1+tanh(x/2))` | 11% |


## bf16

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.exp2(x)` | **`tl.exp(x * LN2)`** | **33%**  |
| `tl.sqrt(x)` | **`x * tl.rsqrt(x)`** 或 `1.0/tl.rsqrt(x)` / `tl.sqrt_rn(x)` | **27%-31%** |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 31%  |
| `tl.maximum(x, 0)` 实现 relu | **`tl.where(x>0, x, 0)`** 或 `(x + tl.abs(x)) * 0.5` | **21%** |
| `1 - 2/(tl.exp(2*x)+1)` 实现 tanh | **`tl.tanh(x)`** 或 `(exp(x)-exp(-x))/(exp(x)+exp(-x))` / `2*sigmoid(2x)-1` | 8% |
| `tl.abs(x)` | **`tl.where(x>=0, x, -x)`** 或 `tl.maximum(x, -x)` | **12%** |
| `tl.sigmoid(x)` | `exp(x)/(1+exp(x))` 或 `0.5*(1+tanh(x/2))` 或 `1/(1+tl.exp(-x))` | 9% |
