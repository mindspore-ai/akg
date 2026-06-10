---
name: catlass-matmul-optimization
description: "CATLASS Gemm 性能调优：DispatchPolicy、Tile 与分核负载均衡、Swizzle、何时用 padding/Split-K/Preload。面向 AR 修改 catlass_kernel.asc 中的类型别名。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "matmul"
---

# CATLASS 矩阵乘调优

在 AR 中优先改 **`catlass_op/kernel/catlass_kernel.asc`** 里的 `DispatchPolicy`、`L1TileShape`、`L0TileShape`、`BlockScheduler`（Swizzle）。改完必须能通过编译，且 verify 精度与 profile 指标才有意义。

## 1. 先选对 kernel 族（逻辑条件，不是背编号）

| 条件 | 方向 |
|------|------|
| 常规对齐 Gemm，无复杂尾处理 | `BasicMatmul` + Pingpong |
| 需要 D=A@B+X、逐元素融合等 | `MatmulEpilogue` 或 EVG 路径（见 epilogue skill） |
| 内轴未 512B 对齐 | padding 族 kernel，不要只在错误模板上拧 Tile |
| K 很大、单核沿 K 吃不满且 **确实启用多段 K 归约** | Split-K 族；若配置导致 **split 因子为 1**，则与普适 Gemm 等价，换族无收益 |
| 超大 M/N、带宽瓶颈 | 可试 Preload + shuffleK |
| M 或 N 极小、块数远小于核数 | **减小 L1 的 m1/n1** 提高块数，再调 Swizzle |

pipeline 绑定 example 时，用 **reference 的 M/N/K/layout/尾处理** 对照上表，而不是按 benchmark 题号选样例。

## 2. DispatchPolicy

```cpp
// 基线
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

// 大矩阵、减少搬运气泡时可试
using DispatchPolicy = Gemm::MmadAtlasA2Preload<true, true>;
```

注意：Preload 会引入额外流水与 scalar/vector 开销，**小矩阵可能变慢**；应以 profile 为准。

## 3. Tile 与 AIC 负载均衡

基本块数：

```
blocks = ceilDiv(M, m1) * ceilDiv(N, n1)
```

目标：`blocks` 接近 **AIC 核数** 的整数倍，避免少量核承担绝大多数块。

调参顺序建议：

1. 用 `catlass-hardware-constraints` 保证 L1/L0 不溢出  
2. 在可行范围内调整 `m1`、`n1`（保持 `m0,m1` 与 `n0,n1` 关系）  
3. 若 `blocks` 仍远小于核数，**减小** `m1` 或 `n1`，不要一味增大 Tile  
4. 再微调 Swizzle 的 `offset` / `direction`

## 4. Swizzle

`GemmIdentityBlockSwizzle<offset, direction>`：

| 情况 | 起点 |
|------|------|
| M > N，A/B 为 RowMajor | direction = 0 |
| M < N，A/B 为 RowMajor | direction = 1 |

在 direction 定好后，可小步调 `offset`（如 3→4→5）观察 profile；每次只改一项便于 settle 对比。

## 5. 与 PyTorch 路径相关的性能

AR 的 `latency_us` 一般是 **`ModelNew.forward` 端到端**（含 dtype 转换、`npu_format_cast`、算子调用）。若 profile 显示耗时在 Transdata 而非 MMAD：

- 仅改 `.asc` Tile **可能几乎不动指标**
- 需改 `catlass_op/src/catlass_torch.cpp`，减少重复的 `npu_format_cast` 或多余拷贝（须在 `editable_files` 中列出该路径）

## 6. 常见误区

- **只改 `kernel.py` 符号名** 或 Python 逻辑，不碰 Tile → 性能通常不变  
- **Split-K 名不副实**：K 很短或 split=1 时应用 `BasicMatmul`  
- **Tile 过大**：块数少于核数 → 大量核空闲  
- **忽略对齐**：未 padding 硬算 → 精度或性能异常  
- **超过 L1**：编译失败或 silently 选错配置 — 先算容量再提交 eval
