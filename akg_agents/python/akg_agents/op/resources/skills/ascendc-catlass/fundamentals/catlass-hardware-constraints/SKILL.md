---
name: catlass-hardware-constraints
description: "CATLASS TileShape 与 on-chip 缓存容量约束：L1/L0A/L0B/L0C 预算公式、fp16/fp32 Pingpong 双缓冲、512B 对齐与排布对 Tile 选型的影响。调参前必读。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "matmul, conv"
---

# CATLASS 硬件约束与 Tile 选型

改 `L1TileShape` / `L0TileShape` 前，先算缓冲区是否超限；编译期 `static_assert` 失败多半是该问题。

## 片上缓存（Atlas A2 量级，Pingpong STAGES=2）

| 缓冲 | 容量 | 用途 |
|------|------|------|
| L1 | 512 KB | A/B tile 双缓冲 |
| L0A | 64 KB | A tile 双缓冲 |
| L0B | 64 KB | B tile 双缓冲 |
| L0C | 128 KB | C 累加（fp32 累加时按 4 字节/元素） |

Atlas A5 等代际容量可能不同，以当前 `ArchTag` 与仓内文档为准；**不要**把 A2 公式硬套到未验证的架构。

## fp16 + Pingpong（元素个数预算）

记 `L1 = (m1, n1, k1)`，`L0 = (m0, n0, k0)`，元素类型宽 `es`（fp16 为 2），累加宽 `ac`（fp32 为 4），双缓冲乘 **2**：

```
L1 元素量:  m1*k1 + n1*k1          （乘 es*2 得字节，须 ≤ 512KB）
L0A 元素量: m0*k0                  （× es*2 ≤ 64KB）
L0B 元素量: k0*n0                  （× es*2 ≤ 64KB）
L0C 元素量: m0*n0                  （× ac ≤ 128KB，fp32 累加）
```

**示例（仅说明算法）**：`L1=(128,256,256)`、`L0=(128,256,64)`、fp16、Pingpong  
→ L1 字节约 `128*256*4 + 256*256*4`，需逐项代入验证是否低于 512KB。

## fp32 输入时

L1 上 A/B 常按 4 字节计；`L0C` 仍可能按 fp32 累加。fp32 场景往往要把 `k1` 或 `m1/n1` 压低，否则 L1 先触顶。

## 选型原则（与具体 benchmark 编号无关）

1. **M/N/K 对齐**：优先 16 的倍数；RowMajor 下常还要求 **512B 对齐**（fp16 下约 256 元素一行宽），否则应走 padding 类 kernel 思路，而不是硬拧 Tile。
2. **L0 与 L1 关系**：常取 `m0=m1`、`n0=n1`，`k0≤k1`；`k0 = k1/4` 是常见起点，不是唯一解。
3. **先过容量，再谈性能**：任意增大 `k1` 前都要重算 L1；`k1` 小于问题 K 时，外层 K 循环次数会增加。
4. **排布**：RowMajor/ColumnMajor 组合不同，宜优先保证 **L1 上 256 倍数** 的搬运效率；zN 等格式对 256 对齐敏感度较低。
5. **小 M 或极小 K**：Tile 过大 → 基本块数过少、核利用率差；应 **减小 m1/n1 或调整 k1**，并配合 Swizzle，而不是只换 example 名字。

## 512B 对齐（RowMajor）

- 关注矩阵 **内轴**（如 RowMajor 的列方向）是否 512B 对齐
- 未对齐：用 padding 语义扩 shape，或在 pipeline 阶段选用带 padding 的 example 族，而不是在错误 kernel 上只改 Tile

## 排布与推荐 Tile 形状（起点，需再验容量与负载）

| A | B | L1 起点（需验算） | 说明 |
|---|---|------------------|------|
| RowMajor | RowMajor | (128, 256, 256) | 通用 |
| RowMajor | ColumnMajor | (128, 256, 256) | N 连续搬运友好 |
| ColumnMajor | ColumnMajor | (256, 128, 256) | M 方向优先 |