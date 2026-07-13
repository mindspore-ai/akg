# Reduction 类算子核心算法路由

> **⚠️ 使用方式**：先看算法选择对照表确定适用算法，再按链接读取对应详细文档。

---

## 算法选择对照表

| 条件 | 推荐算法 | 原因 | 典型算子 | 详细文档 |
|------|---------|------|---------|---------|
| FullLoad | 直接顺序计算 | 数据整块驻留 UB，CopyIn 一次 | reduce_sum/max, softmax_v2 | — |
| FullLoad + 两次顺序归约 | TwoPass | 第一遍求 A，第二遍用 A 求 B | reduce_var/std, layer_norm | — |
| 分载 + 两次相关归约 | Welford Online | 单遍流式，省一轮 IO | reduce_var/std | [alg-welford.md](alg-welford.md) |
| 分载 + 流式累积误差大 | Welford + Group(8) | 每 8 块合并防误差累积 | reduce_var/std | [alg-welford.md](alg-welford.md) |
| 分载 + 单核处理不完 R 且 A 小 | Group Reduce | 跨核分 R，workspace 同步 | arg_max, reduce_var | [alg-group-reduce.md](alg-group-reduce.md) |
| Sum 精度敏感 | 二分累加 | 相近量级先加，解决大数吃小数 | reduce_sum, reduce_var | [alg-dichotomy.md](alg-dichotomy.md) |
| 分载 + 归约带索引跟踪 | 分片合并 | 逐片归约 + 跨片更新全局索引 | arg_max_v2 | [with-index.md](with-index.md) |

---

## 算法摘要

### Welford Online（在线单遍）

分载模式下流式计算两个相关统计量。单遍扫描，增量更新，支持 Group 化并行合并。详见 [alg-welford.md](alg-welford.md)。

### Group Reduce（跨核归约）

R 太大单核处理不完，同时 A 太小不足以多核并行。把 R 分给多个核，各核独立归约后 workspace 同步合并。详见 [alg-group-reduce.md](alg-group-reduce.md)。

### 二分累加 / Half-Interval

Sum 归约专用精度优化。二叉树折叠求和，使相近量级先加。详见 [alg-dichotomy.md](alg-dichotomy.md)。

---
