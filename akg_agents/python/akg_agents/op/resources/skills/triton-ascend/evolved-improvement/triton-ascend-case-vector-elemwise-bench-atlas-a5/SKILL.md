---
name: triton-ascend-case-vector-elemwise-bench-atlas-a5
description: "Atlas A5 上 Triton vector 一元/二元 `tl.*`：fp32/fp16/bf16 三种 dtype 下各算子端到端 time (ms)；并给出\"语义等价、精度对齐\"前提下应替换的 triton API 与推荐写法。"
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
---


# 互替换 API 推荐（Atlas A5）

> **精度告警**：以下替换在单算子单元测试上是"语义等价、精度对齐"的，但在算子融合或者网络中使用时，
> 误差可能累积放大。
> **若替换某条 API 后精度无法对齐，请保留原写法，不要替换。**

## fp32

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `x / (1 + tl.exp(-x))` 实现 silu | **`x * tl.sigmoid(x)`** | **128%** |
| `tl.fdiv(x, y)` / `tl.div_rn(x, y)` / `x*(1.0/y)` 实现 div | **`x / y`**（A5 fp32 上直接除最快） | 87%-94% |
| `x * tl.rsqrt(x)` 实现 sqrt | **`tl.sqrt(x)`** | **71%** |
| `tl.sqrt_rn(x)` | `tl.sqrt(x)` | 55% |
| `tl.where(x < y, x, y)` 实现 minimum | `tl.minimum(x, y)` | 75% |
| `tl.where(x > y, x, y)` 实现 maximum | `tl.maximum(x, y)` | 53% |
| `tl.where(x>=0, x, -x)` 实现 abs | `tl.maximum(x, -x)` 或 `tl.abs(x)` | 32% |
| `1.0/tl.rsqrt(x)` 实现 sqrt | `tl.sqrt(x)` | 27% |
| `exp(x)/(1+exp(x))` 实现 sigmoid | `0.5*(1+tanh(x/2))` 或 `tl.sigmoid(x)` | 23% |
| `tl.exp2(x*LOG2E)` 实现 exp | `tl.exp(x)` | 11% |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 10% |

## fp16

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.sqrt(x)` | **`1.0 / tl.rsqrt(x)`** | **84%-89%** |
| `tl.sigmoid(x)` | **`1/(1+tl.exp(-x))`** | **50%** |
| `tl.where(x > y, x, y)` 实现 maximum | **`tl.maximum(x, y)`** | **59%** |
| `(exp(x)-exp(-x))/(exp(x)+exp(-x))` / `2*sigmoid(2x)-1` 实现 tanh | **`1 - 2/(tl.exp(2*x)+1)`** | **46%-50%** |
| `tl.exp2(x)` | `tl.exp(x*LN2)` | 65% |
| `x / y` 实现 div | `tl.div_rn(x, y)` 或 `tl.fdiv(x, y)` | 25% |
| `tl.maximum(x, 0)` / `tl.where(x>0, x, 0)` 实现 relu | `(x + tl.abs(x)) * 0.5` | 6-11% |
| `tl.where(x>=0, x, -x)` / `tl.abs(x)` | `tl.maximum(x, -x)` | 11-15% |

## bf16

| 原写法 | 推荐替换 | 收益 |
|---|---|---|
| `tl.where(x < y, x, y)` 实现 minimum | **`tl.minimum(x, y)`** | **199%** |
| `tl.where(x > y, x, y)` 实现 maximum | **`(x+y+tl.abs(x-y))*0.5`** 或 `tl.maximum(x, y)` | 75% |
| `tl.rsqrt(x)` | **`1.0 / tl.sqrt(x)`** | **73%** |
| `tl.sqrt_rn(x)` / `x*tl.rsqrt(x)` 实现 sqrt | **`1.0 / tl.rsqrt(x)`** 或 `tl.sqrt(x)` | 93%-101% |
| `tl.sigmoid(x)` | **`exp(x)/(1+exp(x))`** 或 `0.5*(1+tanh(x/2))` | **63%** |
| `tl.div_rn(x, y)` | `x / y` | 17% |
| `tl.log(x)` | `tl.log2(x) * LN2` | 14% |
| `x / (1 + tl.exp(-x))` 实现 silu | `x * tl.sigmoid(x)` | 29% |


## 跨 dtype 的二维/特殊场景

| 原写法 | 推荐替换 | 备注 |
|---|---|---|
| `acc / l[:, None]` 或 `acc*(1.0/l)[:, None]` | **`l_recip = 1.0/l; acc * l_recip[:, None]`** | A5 fp32 上 `l_recip+vmul` 最快，原版直接除慢 **46%**、内联 recip 慢 74% |
