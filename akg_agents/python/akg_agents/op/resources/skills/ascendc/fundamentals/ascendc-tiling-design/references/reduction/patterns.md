# Reduction 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。

---

## 合轴

将 N 维 shape + axes 简化为更少维度。每个维度标记为 **A**（保留轴）或 **R**（归约轴），然后：
1. **消除冗余维度**：size=1 且内存连续的维度消除
2. **合并相邻同类型轴**：相邻维度都是 A（或都是 R）→ 合并（乘积）

**示例**：
- shape=[2,100,4], axes=[1,2] → 标记 A,R,R → 相邻 R 合并 → [2, 400] = (A, R) → 单轴 AR
- shape=[2,3,4,5,6,7,8], axes=[1,2,4] → 标记 A,R,R,A,R,A,A → 相邻 R 合并 → [2,12,5,6,56] = (A,R,A,R,A) → 多轴归约

---

## 场景判定流程

```
给定: shape, axes (归约轴)

Step 0: 合轴（见上节）
  标记 A/R → 消除冗余维 → 合并相邻同类型轴

Step 1: 合轴后是单轴还是多轴？
  ├─ 单轴（AR 或 ARA）→ Step 2
  └─ 多轴（ARAR 等交替序列）→ Shape 三步变换（详见 multi-axis-transform.md）
       展开为嵌套循环，每个 R 维的归约按 Step 2 判定

Step 2: A0 决定模式
  ├─ A0 = 1 → AR 模式（每行 R 个元素连续，用 Level 2 Reduce API）
  │   ├─ 可以在UB中至少处理1整行数据 → AR-FullLoad    → [ar-fullload.md]
  │   └─ 否则 → AR-ColSplit    → [ar-colsplit.md]
  │
  └─ A0 > 1 → ARA 模式（[R,A0] 块连续，用 Pattern::Reduce::RA）
      ├─ 可以在UB中处理所有R*tileA0Len数据（32字节对齐） → ARA-FullLoad  → [ara-fullload.md]
      └─ 否则 → ARA-RowSplit  → [ara-rowsplit.md]

常见特例（本质都是 AR 或 ARA，按上面走）：
  - axis=0（最外维）→ A1=1, ARA 模式
  - 全轴归约        → 所有维度合并为 R, A0=1, AR 模式
  - Norm 类（归约+广播变换）→ 归约部分走 AR/ARA，变换部分是应用层逻辑
```

**正交维度**（确定分支后，根据算子特征和数据规模选择）：

| 维度 | 选项 | 适用条件 | 详细文档 |
|------|------|---------|---------|
| 算法选择 | Welford Online | 分载 + 需要流式计算两个相关统计量 | [algorithms.md](algorithms.md#2-welford-online-算法在线单遍) |
| 多核策略 | Group Reduce | R 太大单核处理不完 + A 太小不足以并行 | [algorithms.md](algorithms.md#3-group-reduce跨核归约) |
| 精度策略 | 二分累加 | 大向量 sum 精度敏感 | [algorithms.md](algorithms.md#5-二分累加dichotomy-addition) |
| 索引跟踪 | With-Index | 归约+返回极值位置 | [with-index.md](with-index.md) |

---

## 通用规则

以下规则适用于所有场景。Tiling 设计原则和 tmpBufSize 公式见 [tiling-fields.md](tiling-fields.md)。

**rLength vs rLengthAlign 参数使用对照表**:

| 参数位置 | 用 rLength（有效数据） | 用 rLengthAlign（对齐后） |
|---------|:---:|:---:|
| DataCopyPad blockLen | ✅ | ❌ |
| Reduce API count（Level 2） | ✅ | ❌ |
| UB 内 rowOffset 计算 | ❌ | ✅ |
| Buffer 大小分配 | ❌ | ✅ |

---

## S1: AR 模式（最内维归约）

适用: 合轴后单轴归约，且归约轴式尾轴的场景

每行 R 个元素连续，使用 Level 2 Reduce API 逐行归约。

**AR 分支决策树**：

```
AR 模式 (A1, R)，A0 = 1
    │
    ├─ 可以在UB中至少处理1整行数据？
    │   │
    │   YES → AR-FullLoad（全载）
    │   │     整行 R 个元素驻留 UB，CopyIn 一次完成归约
    │   │     中间结果直接复用，不需要重复搬入
    │   │
    │   NO  → AR-ColSplit（分载）
    │         对列方向分段加载归约，每次搬入 Ar < AR 个元素
    │         需要跨 chunk 合并（Max/Add）
    │
    └─ 两种模式都使用 Level 2 Reduce API（数据连续，无需 Pattern）
```

| 分支 | 条件 | 说明 | 详细文档 |
|------|------|------|---------|
| **全载（FullLoad）** | 可以在UB中至少处理1整行数据 | 整行驻留 UB，中间结果直接复用 | [ar-fullload.md](ar-fullload.md) |
| **分载（ColSplit）** | 其他 | 对列方向分段，每次搬入 Ar < AR 个元素，跨 chunk 合并 | [ar-colsplit.md](ar-colsplit.md) |

> **索引变体**：需要返回极值位置时，AR-FullLoad 用 `ReduceMax(calIndex=true)`，详见 [with-index.md](with-index.md)。

> **AR 归约后的标量广播操作**：Level 2 Reduce 归约 [R] 后得到 1 个标量。
> 如果后续需要将该标量广播到 [R] 向量参与逐元素运算，使用 `Adds(dst, src, scalar, count)`
> 或 `Muls(dst, src, scalar, count)`，一次 API 调用完成，替换 Duplicate 填充 + Add/Mul操作。
> 详见 `/ascendc-api-best-practices` 的 `api-arithmetic.md`。

---

## S2: ARA模式

适用: 合轴后单轴归约，且归约轴非尾轴场景

> **核心认知**：ARA 模式 `(A1, R, A0)` 下，沿外层 A1 切分后，每次处理一个 `[R, A0_inner]` 块。
> 该块在 GM 中连续，整块搬入 UB 后即为 `(R, alignedCols)` 的二维矩阵，

**数据流**:
```
GM (A1, R, A0) → DataCopyPad(blockCount=R) → UB (R × alignedCols)
    ↓
[ReduceMax/ReduceSum Pattern::Reduce::RA, srcShape={R, alignedCols}] → result (alignedCols)
    ↓
UB (alignedCols) → GM (A1, A0)
```

**关键 API 调用**:
```cpp
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t srcShape[] = {R, alignedCols};
ReduceMax<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);
```

**ARA 分支决策树**：

```
ARA 模式 (A1, R, A0)，A0 > 1
    │
    ├─ 可以在UB中处理所有R*tileA0Len数据（32字节对齐）？
    │   │
    │   YES → ARA-FullLoad（全载）
    │   │     [R, tileA0Len] 整块驻留 UB，CopyIn 一次完成全部 R 行的归约
    │   │     中间结果直接复用，不需要重复搬入
    │   │
    │   NO  → ARA-RowSplit（分载）
    │         每次搬入 r < R 行，分多次才能完成全部 R 行的归约
    │         需要跨 chunk 合并（Max/Add）
    │
    └─ 两种模式都使用 Pattern::Reduce::RA（不需要 Transpose）
```
- tileA0Len 是沿着A0进行多核切分后每个核处理的a0(<=A0)

| 分支 | 条件 | 说明 | 详细文档 |
|------|------|------|---------|
| **全载（FullLoad）** | 可以在UB中处理所有R*tileA0Len数据（32字节对齐） | R 行一次放入 UB，中间结果直接复用 | [ara-fullload.md](ara-fullload.md) |
| **分载（RowSplit）** | 其他 | 每次搬入 r < R 行，分多次归约后跨 chunk 合并 | [ara-rowsplit.md](ara-rowsplit.md) |

> **索引变体**：需要返回极值位置时，用 `Compare+Select` 逐行迭代替代 `Pattern::Reduce::RA`。
> 详见 [with-index.md](with-index.md)。

> **ARA 归约后的广播操作**：Pattern::Reduce::RA 归约 [R, alignedCols] 后得到 [1, alignedCols] 的结果向量。
> 如果后续需要将该向量广播回 [R, alignedCols] 参与逐元素运算，使用 Sub/Div/Mul 等二元 API 的
> BinaryRepeatParams 版本，设置 `src1RepStride=0`，一次 API 调用完成所有 R 行，
> 无需手动循环或额外 broadcast buffer。详见 `/ascendc-api-best-practices` 的 `api-arithmetic.md`。

### S3: 多轴归约

适用: 合轴后多轴归约，比如：ARARA场景

**Shape 变换**：任意 N 维 shape + axes 经三步变换压缩为 A/R 交替序列。
详见 [multi-axis-transform.md](multi-axis-transform.md)。

### S4: Welford Online（R 需 UB 切片）

**适用**: ARA 模式下需要流式计算两个相关统计量（第二个依赖第一个的增量更新）。典型算子：reduce_var / reduce_std。参见 [algorithms.md](algorithms.md#2-welford-online-算法在线单遍)。

### S5: Group Reduce（R 跨核）

**条件**: R 太大，单核无法完成全部 R 归约；同时 A 太小不足以充分利用多核

```
Phase1（各核独立）: 每核处理 A[段] × R[段]，输出 partial → workspace
SyncAll()
Phase2（合并）: 遍历各核 partial，合并为最终结果
```

Workspace: `coreNum × CeilAlign(outAAlign × 2 × sizeof(int32_t), 256)`

---

## S6: 全局归约

适用: reduce_sum(axes=所有轴), reduce_max(axes=所有轴)

所有元素按核均分，各核独立归约后两阶段合并（Atomic 或显式 Merge）。

```
Stage1: 各核 ReduceSum(mySlice) → partial → workspace[blockIdx * 64B]
Stage2:
  方式A: SetAtomicAdd → DataCopy → SetAtomicNone → SyncAll
  方式B: SyncAll → core0 遍历 workspace 合并
```

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| 多输出 Buffer 方程 | [multi-output-buffer.md](multi-output-buffer.md) |
| 通用 Tiling 字段定义 | [tiling-fields.md](tiling-fields.md) |
| 索引跟踪变体 | [with-index.md](with-index.md) |
