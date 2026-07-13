---
name: ascendc-perf-optimize
description: Ascend C 算子性能优化知识库。基于 profiling / 仿真图判定 bound 类型（VEC / 访存 / Scalar / 无 bound），分类加载对应优化策略；含参数空间分析。触发：算子性能调优、流水分析、bound 诊断、tiling 参数搜索时。
---

# Ascend C 算子性能优化知识

## Bound 类型与优化策略

按 profiling / 仿真图的瓶颈类型路由：

| Bound | 判定条件 | 文档 |
|---|---|---|
| **VEC bound** | Vector 单元利用率高、向量指令主导耗时 | [vec.md](references/single-core-pipeline/vec.md) |
| **访存 bound** | MTE2/AIC 带宽接近峰值、Vector 等数据 | [memory.md](references/single-core-pipeline/memory.md) |
| **Scalar bound** | scalar 耗时占比高、控制流密集或 shape 很小 | [scalar.md](references/single-core-pipeline/scalar.md) |
| **无 bound** | 各单元利用率都不高，仍需进一步榨油 | [no-bound.md](references/single-core-pipeline/no-bound.md) |

每篇文档覆盖：判定条件、严重程度分级、仿真图 trace 特征、具体策略（融合指令、DoubleBuffer、Cast 优化、L2 复用等）、tiling 修正建议。

若单一 bound 不明显，或只有少数样本极慢，先阅读 `ascendc-profiling-optimization` 的“样本反馈与窄路径”和“常见可复用优化范式”。这类问题通常不是继续调一个数字参数，而是需要按 dtype、rank、broadcast、reduce axis、index 连续性或特殊数值模式拆分路径。

## Tiling 参数空间分析

设计调优搜索空间前必读：[parameter-analysis.md](references/tiling/parameter-analysis.md) —— 四阶段方法（kernel 参数全集 → 类型/约束追溯 → 算法使能判定 → 候选解空间）。
