# 多轴归约处理

> 本文档描述合轴后仍为多段 A/R 交替序列（如 ARAR）时，如何展开执行。
> 合轴规则见 [patterns.md](patterns.md#合轴)。

---

## 实际算子示例

| 算子 | Shape | Axes | 合轴后 Pattern |
|------|-------|------|---------------|
| reduce_sum | [2,3,4,5] | [1,3] | **ARAR** [2,3,4,5] |
| reduce_sum | [2048,2,48,2,2,2] | [1,3,5] | **ARARAR** [2048,2,48,2,2,2] → pad→ **ARARARAR** (8维) |
| bn_training_reduce | [N,C,H,W] | [0,2,3] | **ARAR** [1,N,C,H×W] |

## 典型场景：BatchNorm "保留 C、归约 N/H/W"

```
bn_training_reduce: 对 NCHW 张量沿 N,H,W 归约，保留 C 通道维
  输入: [N, C, H, W]    axes=[0, 2, 3]
         R  A  R  R

  合轴: axes 2,3 相邻且同为 R → 合并 → R[N], A[C], R[H×W]
  前置 A[1] → ARAR [1, N, C, H×W]

  输出: sum[C] 和 squareSum[C]（双输出，每个通道一个值）
  特点: 双输出归约 — 同一遍扫描同时算 Σx 和 Σx²，避免二次遍历
```

---

## 高维 Pattern（≥5 维）的统一处理

变换后维度 ≥ 5 时，通过 PadDimOne() 填充 size=1 维度统一到 **8 维（ARARARAR）或 9 维（ARARARARA）**：

```
5 维 ARARA → pad 到 9 维 ARARARARA
6 维 ARARAR → pad 到 8 维 ARARARAR
7 维 ARARARA → pad 到 9 维 ARARARARA
8 维 → 直接 ARARARAR
9 维 → 直接 ARARARARA
```

---

## Kernel 侧执行方式

多轴 Pattern 展开为嵌套循环，每层 R 轴独立走 AR 或 ARA 判定：

```
IterateInnerA<0, N>()   ← 遍历所有 A 轴（递归模板，编译期展开）
  for a0 in A_axis_0:
    for a1 in A_axis_1:
      ...
        LinearComputeR()  ← 处理对应 R 轴归约
          for r in R_axis:
            CopyIn → PreReduce → ReduceCompute → DoCaching → PostReduce → CopyOut
```

**每层 R 轴的 AR/ARA 判定**：看该 R 轴右侧是否还有 A 维度
- R 轴是最内层（右侧无 A）→ **AR 模式**，用 Level 2 Reduce API，详见 [ar-fullload.md](ar-fullload.md) / [ar-colsplit.md](ar-colsplit.md)
- R 轴右侧还有 A 维度 → **ARA 模式**，用 Pattern::Reduce::RA，详见 [ara-fullload.md](ara-fullload.md) / [ara-rowsplit.md](ara-rowsplit.md)

**示例**：ARAR [2,3,4,5] 展开为：
```
for a0 in range(2):       ← A 轴 0
  for r0 in range(3):     ← R 轴 0（右侧有 A[4] → ARA 模式）
    for a1 in range(4):   ← A 轴 1
      for r1 in range(5): ← R 轴 1（右侧无 A → AR 模式）
        Reduce(tile)
```

---

## 非连续多轴的数据搬运

多轴归约时内存往往不连续（如 axes=[0,2] 使得 R 轴散布在内存中）：

```
DAV_3510:    CopyInWithNddma() — 多维 DMA 自动处理 stride 跳跃
DAV_2201: DataCopyPad 的 blockCount/blockLen/srcStride 参数配置 stride copy
       或外层循环逐 slice 搬运（每次搬连续片段，循环处理不连续间隔）
```
