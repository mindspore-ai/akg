---
name: ascendc-performance-best-practices
description: Ascend C 算子性能优化最佳实践库。按算子族组织优化设计文档（softmax / elementwise / broadcast / scalar / common），用于实施阶段查询参考实现。触发：实施某项优化时需要参考代码 / 设计模板。
---

# Ascend C 算子性能优化最佳实践

按 **算子族（operator family）** 组织优化知识。

## 算子族设计文档

| 算子族 | 典型算子 | 设计文档 |
|---|---|---|
| **Reduction / Softmax** | Softmax, log_softmax, FlashAttention 内嵌 | [online_softmax_design.md](reference/softmax/online_softmax_design.md) (FlashAttention-style running max+sum), [state_resident_design.md](reference/softmax/state_resident_design.md) |
| **Elementwise** | Sin, Cos, Abs, Exp, sigmoid, tanh | [double_buffer_design.md](reference/elementwise/double_buffer_design.md), [vector_efficiency_design.md](reference/elementwise/vector_efficiency_design.md) |
| **Broadcast** | Add, Mul, Sub 含广播语义 | [broadcast_mask_design.md](reference/broadcast/broadcast_mask_design.md) |
| **Scalar 编码** | ScalarBound 类（控制流 / 索引密集） | [guide.md](reference/scalar/guide.md), [coding_principles.md](reference/scalar/coding_principles.md) |

## 跨算子族通用模式

| 优化类型 | 适用场景 | 文档 |
|---|---|---|
| **尾块处理** | 数据量不能被 tile 大小整除 | [tail_block_design.md](reference/common/tail_block_design.md) |
| **DataCopy 优化** | 非对齐、小批量多次搬运 | [datacopy_optimization_design.md](reference/common/datacopy_optimization_design.md) |
| **UB / TBuf 常驻复用 & Bank 冲突规避** | 大量 tile/loop 都重复从 GM 搬运同份数据 | [ub_resident_design.md](reference/common/ub_resident_design.md) |

## 每份 design.md 的章节约定

- 优化目标（量化收益）
- 架构概览（存储层级 / 数据流 / 事件同步）
- 关键参数（host 侧计算与字段）
- 核心计算循环（改造前后对照）
- 关键修改点表格
- 可选：约束 / 踩坑 / 选型决策
