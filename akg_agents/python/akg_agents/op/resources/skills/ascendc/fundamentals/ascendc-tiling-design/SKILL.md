---
name: ascendc-tiling-design
description: Ascend C 算子 Tiling 设计指南。覆盖 Reduction / Elementwise / Broadcast 三类算子的场景路由、算法选择、UB 切分公式、tiling 字段约定。触发：算子设计阶段、规划多核切分 / UB 切分、Buffer 分配、查阅某类算子的 Tiling 方法论。
---

# Ascend C 算子 Tiling 设计

## 算子族场景路由

每族先读 `patterns.md` 做场景判定，再按结论进入具体算法文档。

| 族 | 典型算子 | 场景路由 |
|---|---|---|
| **Reduction** | ReduceSum, Softmax, LayerNorm, ArgMax, RMSNorm | [reduction/patterns.md](references/reduction/patterns.md) |
| **Elementwise** | Sin, Cos, Abs, Exp, sigmoid, mish, gelu | [elewise/patterns.md](references/elewise/patterns.md) → [elewise/tiling.md](references/elewise/tiling.md) |
| **Broadcast** | Add, Mul, Sub 含广播语义 | [broadcast/patterns.md](references/broadcast/patterns.md) |

## Reduction 算法库

| 算法 | 适用 | 详细 |
|---|---|---|
| **FullLoad（直算）** | 数据整块驻 UB，CopyIn 一次 | (内联在 patterns.md) |
| **TwoPass** | FullLoad 下两次顺序归约（var/std/layer_norm） | (同上) |
| **Welford Online** | 分载，单遍流式两个相关统计量 | [alg-welford.md](references/reduction/alg-welford.md) |
| **Welford + Group(8)** | 分载流式累积误差大时 | (同 alg-welford.md) |
| **Group Reduce** | 跨核归约（R 大单核装不下、A 小） | [alg-group-reduce.md](references/reduction/alg-group-reduce.md) |
| **二分累加 / Dichotomy** | Sum 精度敏感，解决大数吃小数 | [alg-dichotomy.md](references/reduction/alg-dichotomy.md) |
| **带索引归约（ArgMax/Min）** | 归约 + 索引跟踪 | [with-index.md](references/reduction/with-index.md) |

AR / ARA 模式（按合轴后的轴序）：

| 模式 | 触发 | 子模式 |
|---|---|---|
| AR-FullLoad | A0=1，UB 装得下一整行 | [ar-fullload.md](references/reduction/ar-fullload.md) |
| AR-ColSplit | A0=1，UB 装不下整行 | [ar-colsplit.md](references/reduction/ar-colsplit.md) |
| ARA-FullLoad | A0>1，UB 装得下 R × tileA0 | [ara-fullload.md](references/reduction/ara-fullload.md) |
| ARA-RowSplit | A0>1，UB 装不下 | [ara-rowsplit.md](references/reduction/ara-rowsplit.md) |

其它配套文档：[algorithms.md](references/reduction/algorithms.md)（路由总表）、[multi-axis-transform.md](references/reduction/multi-axis-transform.md)（多轴归约的 shape 三步变换）、[multi-output-buffer.md](references/reduction/multi-output-buffer.md)（多输出 buffer 规划）、[tiling-fields.md](references/reduction/tiling-fields.md)（tiling 结构体字段约定）。

## Broadcast 子方案

| 方案 | 适用 | 详细 |
|---|---|---|
| 一维广播 | shape 简单可展平 | [onedim.md](references/broadcast/onedim.md) |
| UB 内广播（PadBC） | 广播张量小，UB 装得下 | [ub-broadcast.md](references/broadcast/ub-broadcast.md) |
| Dynamic UB 广播 | 广播 shape 运行时确定 | [dynamic-ub-broadcast.md](references/broadcast/dynamic-ub-broadcast.md) |
| NDDMA 广播 | 大 shape 走硬件搬运广播 | [nddma-broadcast.md](references/broadcast/nddma-broadcast.md) |

## 通用 Tiling 设计要素

任何算子都必须考虑：

1. **多核切分**：负载均衡、数据局部性、粒度（每核 ≥ 4KB）
2. **UB 切分**：UB 容量限制（910B/B3 = 192KB，910B4 = 128KB，950 = 248KB），单次处理量，分 chunk 公式
3. **Buffer 规划**：input/output queue、calcBuf 临时、DoubleBuffer × BUFFER_NUM
4. **分支覆盖**：dtype（FP32/FP16/BF16/INT8）、shape 大小、对齐、边界值

详细 UB 字节账本见 [[ascendc-ub-budget]]，UB 上的 LocalTensor 子视图规则见 [[ascendc-localtensor-subviews]]。
