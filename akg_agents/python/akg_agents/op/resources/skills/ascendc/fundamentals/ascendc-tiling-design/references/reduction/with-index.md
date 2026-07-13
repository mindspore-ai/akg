# Reduction with Index Tracking (归约+索引跟踪)

> 索引跟踪是 Reduction 的一个**正交变体**：在归约的同时记录极值（或其他条件）所在位置。
>
> **Tiling 方法论完全相同**：3D 抽象、AR/ARA 判定、全载/分载判定、多核切分 — 全部复用标准 Reduction 分支文档（[ar-fullload.md](ar-fullload.md)、[ara-fullload.md](ara-fullload.md) 等）。
>
> **本文档只描述索引跟踪的增量差异**：API 替换、额外约束、Buffer 增量。

---

## 与标准 Reduction 的差异总结

| Branch | 标准 Reduction | 索引跟踪变体 |
|--------|---------------|-------------|
| AR-FullLoad | `ReduceMax(dst, src, tmp, count)` | `ReduceMax(dst, src, tmp, count, calIndex=true)` |
| AR-ColSplit | chunk → ReduceMax → 标量合并 | chunk → `ArgMaxV1` → 跨片索引偏移合并 |
| ARA-FullLoad | `Pattern::Reduce::RA` → 向量结果 | `Compare(LE) + Select(TT) + Select(TS)` 逐行迭代 |
| ARA-RowSplit | 同 ARA-FL + 跨 chunk `Max` | 同 ARA-FL + 跨 chunk Compare+Select 合并 |

**额外约束**（所有分支共用）：

| 约束 | 说明 |
|------|------|
| **索引类型** | DAV_2201 上 Select 不支持 int32 dst，索引必须用 float 存储，最后 Cast 为 int32 |
| **索引范围** | float32 精确表示 [0, 2^24] 整数；half <= 65535 |
| **多个极值** | 返回**第一个**极值的索引（LE/GE 方案自动保证） |
| **Compare 256 字节对齐** | count 个元素占 256 字节对齐（float: 64 元素倍数），**非** 32 字节 |
| **DataCopyPad rightPadding** | rightPadding <= 32 字节（最多 8 个 float），大 padding 量不能用此参数 |
| **Select 8K 预留** | DAV_2201 上模式 1/2 需预留 8K UB，框架通常自动管理 |

---

## AR 分支的索引变体

> Tiling 参数计算、多核切分、DataCopyPad 配置：参见 [ar-fullload.md](ar-fullload.md) / [ar-colsplit.md](ar-colsplit.md)。
> 以下仅描述索引跟踪在 AR 分支下的 API 差异。

### AR-FullLoad：ReduceMax(calIndex=true)

数据连续，直接用 Level 2 API 的 `calIndex=true` 参数：

```cpp
AscendC::LocalTensor<float> dstVal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> sharedTmpBuffer = tmpQueue.AllocTensor<float>();

AscendC::ReduceMax<float>(dstVal, srcVal, sharedTmpBuffer, count, true);
// calIndex=true 同时返回值和索引

// 输出格式：dst[0]=最大值, dst[1]=索引
float maxVal = dstVal.GetValue(0);
float idxRaw = dstVal.GetValue(1);
uint32_t maxIdx = *reinterpret_cast<uint32_t*>(&idxRaw);  // 类型转换！
```

**tmpBuffer 计算**（calIndex=true 时更大）：

```cpp
uint32_t tmpSize = AscendC::GetReduceMaxMinTmpSize<T>(count, true);
```

**Buffer 规划**（vs 标准 AR-FullLoad 的差异）：

| Buffer | 大小 | 用途 | 标准 Reduction 是否需要 |
|--------|------|------|----------------------|
| srcQueue | count×sizeof(T) | 输入数据 | 相同 |
| dstQueue | **2×sizeof(T)** | 输出值+索引 | 标准只需 1×sizeof(T) |
| tmpQueue | tmpBufSize×sizeof(T) | 中间计算 | 相同，但 calIndex=true 时更大 |

### AR-ColSplit：ArgMaxV1 + 跨片索引合并

当 R 太大无法全载时，分 chunk 处理。每片用 `ArgMaxV1` 独立求局部最大值+索引，跨片合并。

```cpp
// ArgMaxV1: 对连续的 R_slice 个元素，找最大值及其下标
// dst_indice: 输出索引（0-based，相对本片起始位置）
// dst_values: 输出最大值
ArgMaxV1(dst_indice, dst_values, src, batchSize, R_slice);
```

**跨片合并逻辑**：

1. **第一片**：ArgMaxV1 → 初始 (maxValue, maxIndex)
2. **后续每片**：ArgMaxV1 → (chunkValue, chunkIndex)
   - chunkValue > maxValue → 更新 maxValue，maxIndex = chunkIndex + **片起始偏移**
   - 否则保留原值
3. **尾片**：大小可能小于 cutRSize，同样处理

> **关键**：ArgMaxV1 返回的索引是**片内偏移**（从 0 开始），合并时需加上片的全局起始位置。

> **ArgMin 差异**：比较条件从 `>` 改为 `<`。

> **ARA-RowSplit 场景**：分片合并逻辑相同，区别仅在数据搬运（DataCopyPad 的 blockCount=R_chunk 行，带 srcStride）。

---

## ARA 分支的索引变体

> Tiling 参数计算、多核切分、DataCopyPad 配置：参见 [ara-fullload.md](ara-fullload.md) / [ara-rowsplit.md](ara-rowsplit.md)。
> 以下仅描述索引跟踪在 ARA 分支下的 API 差异。

### 核心差异：Compare+Select 替代 Pattern::Reduce::RA

标准 ARA Reduction 用 `Pattern::Reduce::RA` 一次归约整个 `(R, alignedCols)` 矩阵。
索引跟踪无法使用 Pattern API（无索引输出），改用逐行 `Compare+Select` 迭代。

### API 约束（DAV_2201）

| 约束项 | 具体限制 | 影响 |
|--------|---------|------|
| **Select dst 类型** | DAV_2201 仅支持 half/float，**不支持 int32** | 索引必须用 float 存储，输出前 Cast 为 int32 |
| **Compare count 对齐** | count 个元素所占空间必须 **256 字节对齐**（float: 64 元素倍数） | a0Aligned = ceil(A0/64)*64，非 32 字节对齐 |

### 推荐方案：LE 反转 + TENSOR_SCALAR

**核心技巧**：Compare 用 LE（而非 GT）反转 mask 极性，使 bit=1 表示"保留旧值"，从而用 `VSEL_TENSOR_SCALAR_MODE` 将当前行索引作为 scalar 传入 Select，省掉每轮的 `Duplicate` 操作。循环内 3 条指令/轮，5 个 buffer。

**算法逻辑**：

```
Compare(LE): xLocal[r] <= maxLocal
  → bit=1: 当前行不大于旧最大值 → 保留旧值
  → bit=0: 当前行大于旧最大值   → 更新新值

Select(maxLocal, cmpLocal, maxLocal, xLocal[r], TENSOR_TENSOR):
  → bit=1: 保留 maxLocal（旧最大值）
  → bit=0: 取 xLocal[r]（新最大值）

Select(idxLocal, cmpLocal, idxLocal, rowIdxFloat, TENSOR_SCALAR):
  → bit=1: 保留 idxLocal（旧索引 tensor）
  → bit=0: 取 rowIdxFloat（新索引 scalar）
```

**"首个极值"语义保证**：当 `xLocal[r] == maxLocal` 时，LE 成立（bit=1），保留旧索引——与 numpy.argmax 行为一致。

**参考实现**

```cpp
__aicore__ inline void Compute()
{
    AscendC::LocalTensor<float>   xLocal   = inQueueX.DeQue<float>();
    AscendC::LocalTensor<int32_t> yLocal   = outQueueY.AllocTensor<int32_t>();
    AscendC::LocalTensor<float>   maxLocal = maxBuf.Get<float>();
    AscendC::LocalTensor<float>   idxLocal = idxBuf.Get<float>();      // 索引用 float 存储！
    AscendC::LocalTensor<uint8_t> cmpLocal = cmpBuf.Get<uint8_t>();

    // 初始化：第一行作为初始最大值，索引为 0.0f
    AscendC::DataCopy(maxLocal, xLocal, a0Aligned);
    AscendC::Duplicate<float>(idxLocal, 0.0f, a0Aligned);
    AscendC::PipeBarrier<PIPE_ALL>();  // DataCopy(MTE) + Duplicate(V) 跨 pipe

    // LE 反转 + TENSOR_SCALAR 优化循环
    float rowIdxFloat = 1.0f;  // 用 float 累加器避免 aicore 中 uint→float cast
    for (uint32_t r = 1; r < R; r++) {
        AscendC::Compare(cmpLocal, xLocal[r * a0Aligned], maxLocal,
                         AscendC::CMPMODE::LE, a0Aligned);
        AscendC::Select(maxLocal, cmpLocal, maxLocal, xLocal[r * a0Aligned],
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Aligned);
        AscendC::Select(idxLocal, cmpLocal, idxLocal, rowIdxFloat,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Aligned);
        rowIdxFloat = rowIdxFloat + 1.0f;
    }

    // float 索引 → int32 输出
    AscendC::Cast(yLocal, idxLocal, AscendC::RoundMode::CAST_ROUND, a0Aligned);
    outQueueY.EnQue<int32_t>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

### Buffer 规划（5 个 vs 标准 Reduction 的 3 个）

| Buffer | 类型 | 大小 | 用途 | Queue | 标准 Reduction |
|--------|------|------|------|-------|---------------|
| inQueueX | float | R×a0Aligned×4 | 输入数据 | TQue VECIN, InitBuffer num=2 | 相同 |
| outQueueY | int32 | a0Aligned×4 | 输出索引 | TQue VECOUT, InitBuffer num=2 | 输出值（非索引） |
| maxBuf | float | a0Aligned×4 | 当前最大值 | TBuf VECCALC | **新增** |
| idxBuf | float | a0Aligned×4 | 当前索引（float 存储） | TBuf VECCALC | **新增** |
| cmpBuf | uint8_t | max(a0Aligned/8, 32) | 比较结果 mask | TBuf VECCALC | **新增** |

### UB 计算

```
UB_USED = 2 × (R × a0Aligned × 4)       // inQueueX (num=2 开启 Double Buffer)
        + 2 × (a0Aligned × 4)           // outQueueY (num=2 开启 Double Buffer)
        + a0Aligned × 4                  // maxBuf
        + a0Aligned × 4                  // idxBuf
        + max(a0Aligned / 8, 32)         // cmpBuf（32 字节对齐）
```

### Compare 256 字节对齐策略

**对齐计算**（非 32 字节对齐）：

```cpp
// float 类型：向上取整到 64 的倍数（256 字节 / 4 字节 = 64 元素）
uint32_t a0Aligned = ((A0 + 63) / 64) * 64;
```

| 原始 A0 | a0Aligned | Pad 量 |
|---------|-----------|--------|
| 32 | 64 | 32 |
| 36 | 64 | 28 |
| 64 | 64 | 0 |
| 65 | 128 | 63 |

**Pad 区域无需预填充**：CopyOut 只输出前 curA0Len 个有效元素，pad 区域结果不写回 GM。省掉预填充可减少一次大范围矢量写操作和一次 `PipeBarrier<PIPE_ALL>()`。

---

## Min-Index 变体

与 Max-Index 的唯一区别是 Compare 模式反转：

| | Max-Index | Min-Index |
|--|--------|--------|
| Compare 模式 | **LE** | **GE** |

其余逻辑（Select、Buffer、对齐、索引类型）完全相同。

---

## 性能优化技巧

### 技巧1：LE/GE 反转 + TENSOR_SCALAR 模式

**原理**：通过反转 Compare 的比较方向（GT→LE / LT→GE），使 mask 中 bit=1 表示"保留旧值"（多数位置），bit=0 表示"更新新值"（少数位置）。索引更新的 Select 用 `VSEL_TENSOR_SCALAR_MODE`：
- bit=1 → 从 tensor（idxLocal）取值（保留旧索引）
- bit=0 → 从 scalar（rowIdxFloat）取值（更新新索引）

**收益**：省 Duplicate + 省 1 个 buffer + 循环 3 条指令/轮

**适用条件**：ARA 分支的逐行迭代场景，需要 Select 的 TENSOR_SCALAR 模式支持 float。

### 技巧2：float 累加器替代 Cast

aicore 中不支持 `static_cast<float>(uint32_t)`。用 float 变量逐步 +1.0f：

```cpp
float rowIdxFloat = 1.0f;
for (uint32_t r = 1; r < R; r++) {
    // 使用 rowIdxFloat 而非 static_cast<float>(r)
    ...
    rowIdxFloat = rowIdxFloat + 1.0f;
}
```

---
