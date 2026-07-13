---
name: ascendc-batch-playbook
description: "AscendC 24-op 批量开发/批跑 playbook：算子分类、模板复用、失败分层、优先级、日志判读和每轮只改一个假设。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC 批量开发与批跑 Playbook

当一轮需要同时推进多个 AscendC 算子时使用本 skill。目标是减少重复错误，让不同 op 的结论可比较、可复用。

## 1. 单算子建档

每个 op 先记录以下信息，再开始改代码：

```text
op name:
family: elementwise | broadcast | reduction | indexed | matmul-like | fused
input ranks:
output shape rule:
dtypes:
special cases:
expected tolerance:
initial skeleton:
first failing shape:
```

`initial skeleton` 从 `ascendc-op-patterns` 选择。不要在没有分类的情况下直接写 kernel。

## 2. 批跑优先级

推荐推进顺序：

1. 让所有算子能 configure、build、load。
2. 让简单 32B 对齐 shape 正确。
3. 让 tail、非对齐、非整除 shape 正确。
4. 让 dtype 变体正确。
5. 只优化已经通过正确性验证的算子。
6. 对最慢且已正确的算子做 profiling-driven tuning。

仍有未解释精度错误的 kernel 不进入性能优化。

## 3. 失败分桶

| 失败类型 | 优先使用的 skill |
|---|---|
| CMake/configure/build 失败 | `ascendc-direct-invoke`、`ascendc-crash-debug` |
| `.so` 缺失或 op namespace 错误 | `ascendc-direct-invoke` |
| timeout、hang、aic error | `ascendc-crash-debug` |
| 输出全 0、随机值、err_cnt | `ascendc-precision-debug` |
| 仅 tail shape 失败 | `ascendc-hardware-tiling` |
| 结果正确但性能慢 | `ascendc-profiling-optimization` |
| 初始 kernel 结构不确定 | `ascendc-op-patterns` |

## 4. 批量不变量

- `kernel.py` 保持稳定 wrapper 形态：懒加载 `.so`，调用 `torch.ops.npu.<op>`。
- CMake target 名称、`.so` 文件名、注册 op 名称保持一致。
- host tiling 公式和 kernel tiling struct 同步修改。
- build/load 修复与数学逻辑修复分开提交或分轮处理。
- 每轮只保留一个 active failing shape。
- 不静默收缩 dtype、layout 或 shape 覆盖。

## 5. 何时抽象复用

至少两个算子已经证明共享同一模式后，再抽象公共模板：

- elementwise flatten skeleton
- broadcast fast-path skeleton
- row-reduction skeleton
- two-stage reduction skeleton
- softmax-like row skeleton
- indexed fallback skeleton

在 ABI、tiling 字段和输出规则稳定前，不要提前引入通用 helper。

## 6. 每轮最小报告

每轮记录：

```text
hypothesis:
changed files:
tested shape(s):
result:
next action:
```

若同一错误在两次盲改后仍存在，停止继续改代码，转向日志、tiling 值、DumpTensor 或最小 shape 复现。
