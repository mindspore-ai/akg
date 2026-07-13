# Reduce 类算子 - ARA 全载分支

> **适用场景**: A0>1（非尾轴）, R ≤ R_max（全载模式）

---

## 目录

- [一、分支特征](#一分支特征)
- [二、Buffer 规划](#二buffer-规划)
- [三、Tiling 参数计算](#三tiling-参数计算)
- [四、Kernel 实现要点](#四kernel-实现要点)
- [五、常见问题](#五常见问题)
- [六、性能优化建议](#六性能优化建议)

---

## 一、分支特征

| 特征 | 说明 |
|------|------|
| **模板类型** | ARA 模板（A 轴 + R 轴 + A 轴） |
| **Shape 抽象** | (A1, R, A0) |
| **载入模式** | 全载（全部 R 行完整放入 UB） |
| **适用条件** | A0>1, R ≤ R_max |
| **数据连续性** | 每列 R 个元素不连续（间隔 A0） |
| **Reduce 结果** | 向量（A0 个值） |

---

## 二、Buffer 规划

### 2.1 FP32 场景（Kernel侧）

```cpp
// Double Buffer 模式
constexpr uint32_t a0TileBase = 64;  // FP32

pipe->InitBuffer(inQueueX, 2, R * a0TileBase * sizeof(float));
pipe->InitBuffer(outQueueY, 2, a0TileBase * sizeof(float));  // Reduce结果（A0个值）
pipe->InitBuffer(tmpBuf, tmpBufSize);

// 总 UB = 2×R×a0TileBase×4 + 2×a0TileBase×4 + tmpBufSize
```

### 2.2 tmpBufSize 计算（Host侧）

```cpp
constexpr uint32_t VECTOR_REG_WIDTH = 256;
uint32_t perRepeat = VECTOR_REG_WIDTH / sizeof(float);  // 64
uint32_t perBlock = 32 / sizeof(float);  // 8

// tmpBufSize 基于 R × alignedCols
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t repeats = (R * alignedCols + perRepeat - 1) / perRepeat;
uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * sizeof(float);
tmpBufSize = std::max(tmpBufSize, 4096u);
```

---

## 三、Tiling 参数计算

### 3.1 全载阈值计算（Host侧）

```cpp
// 全载条件: 2×R×a0TileBase×4 + 2×a0TileBase×4 + tmpBufSize ≤ UB_SIZE
constexpr uint32_t a0TileBase = 64;  // FP32
uint32_t tmpBufSize = 4096;
uint32_t overhead = 2 * a0TileBase * sizeof(float) + tmpBufSize;
uint32_t R_max = (UB_SIZE - overhead) / (2 * a0TileBase * sizeof(float));

R_max = std::min(R_max, 255u);  // API 限制 repeatTimes ≤ 255
R_max = std::max(R_max, 1u);

// 估算值: 公式计算约 375，但受 min(R_max, 255) 限制，实际 R_max = 255
```

### 3.2 A0Inner 计算（三约束取最小）

```cpp
// 约束1: UB 容量限制
uint64_t ubPerTileBase = 2 * R * a0TileBase * sizeof(float) + tmpBufSize;
uint64_t fixedOverhead = 2 * a0TileBase * sizeof(float) + tmpBufSize;
int64_t factorMax = (UB_SIZE - fixedOverhead) / ubPerTileBase;
if (factorMax < 1) factorMax = 1;

// 约束2: A0 维度限制
int64_t a0FactorMax = (A0 + a0TileBase - 1) / a0TileBase;

// 约束3: 多核负载均衡限制
int64_t totalTilesMax = A1 * a0FactorMax;
int64_t a0InnerMax = (totalTilesMax + blockDim - 1) / blockDim;

// 取最小值
int64_t a0Inner = std::min({a0InnerMax, factorMax, a0FactorMax});
a0Inner = std::max(a0Inner, 1L);

uint32_t tileA0Len = a0Inner * a0TileBase;
```

### 3.3 多核切分参数（Host侧）

```cpp
uint32_t a0Outer = (A0 + tileA0Len - 1) / tileA0Len;
uint32_t totalTiles = A1 * a0Outer;
uint32_t tilesPerCore = (totalTiles + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
uint32_t tailCoreTiles = totalTiles % tilesPerCore;
if (tailCoreTiles == 0 && totalTiles > 0) tailCoreTiles = tilesPerCore;
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM (A1, R, A0) → CopyIn → UB (R × tileA0Len)
    ↓
[ReduceMax/ReduceSum Pattern::Reduce::RA] → result (tileA0Len)
    ↓
UB (tileA0Len) → CopyOut → GM (A1, A0)
```

> **全载模式的核心优势**：[R, tileA0Len] 数据完整驻留在 UB 中。
> 如果算子需要多步计算（如先 ReduceMax 再 ReduceSum），中间结果直接复用 UB 中的数据，
> 只需 CopyIn 一次，不需要重新从 GM 搬入数据（多轮扫描仅在分载模式下才需要，因为每次只能搬入部分数据）。
> 注意：ReduceMax/ReduceSum 的 `isReuseSource=false`（默认值）不破坏源数据，后续步骤可以继续使用。

### 4.2 核心 API 调用

```cpp
// 对齐后的列数（关键！）
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// srcShape 必须使用 alignedCols
uint32_t srcShape[] = {R, alignedCols};

// ReduceMax (Pattern::Reduce::RA - 沿第一维归约)
ReduceMax<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);

// 或 ReduceSum (Pattern::Reduce::RA - 沿第一维归约)
ReduceSum<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);
```

**关键要点**:
- 使用 `Pattern::Reduce::RA` - 沿第一维（R维度）归约
- `srcShape[1]` 必须使用 `alignedCols`（32字节对齐）
- 输出是向量（tileA0Len 个值）

**为什么选择 Pattern::Reduce::RA**:

多核切分后，每个核处理部分 `(R, A0_inner)` 的数据，在 UB 中布局为：
```
[row0的全部A0_inner, row1的全部A0_inner, ..., row{R-1}的全部A0_inner]
```
即 `(R, alignedCols)` 的 2D 矩阵。

Pattern::Reduce::RA 的含义：
- **R** = Reduce 维度（沿第一维归约）
- **A** = Align 维度（保留第二维）

对于每个 `a0` 位置（0 到 A0_inner-1），取 R 个值归约，输出 A0_inner 个结果。

**注意**：不能用 Level 2 API `Reduce<T>(dst, src, tmp, count)`，因为它只能处理连续数据，且只能归约-1轴。

### 4.3 数据搬运（关键！）

**GM→UB 使用 DataCopyPad**:

```cpp
// blockCount=R 行, blockLen=tileA0Len*sizeof(float)
DataCopyExtParams copyParams;
copyParams.blockCount = R;
copyParams.blockLen = tileA0Len * sizeof(float);
copyParams.srcStride = (A0 - tileA0Len) * sizeof(float);  // 跨越 A0
copyParams.dstStride = 0;

DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

**UB→GM 使用 DataCopyPad**:

```cpp
DataCopyExtParams copyParams;
copyParams.blockCount = 1;
copyParams.blockLen = tileA0Len * sizeof(float);
copyParams.srcStride = 0;
copyParams.dstStride = 0;

DataCopyPad(yGm[offset], resultLocal, copyParams);
```

### 4.4 非对齐处理

```cpp
// tileA0Len 可能不是 32 字节对齐
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// Buffer 大小使用 alignedCols
pipe->InitBuffer(inQueueX, 2, R * alignedCols * sizeof(float));

// Reduce API 使用 alignedCols
uint32_t srcShape[] = {R, alignedCols};
```

---

## 五、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 精度过大 | srcShape 使用 A0 而非 alignedCols | 使用对齐后的列数 |
| 编译错误：no matching ReduceMax | API 参数不匹配 | 使用带 Pattern 的接口，srcShape={R, alignedCols} |
| 数据错误 | DataCopy srcStride/dstStride 错误 | 正确计算跨越距离 |
| FP16 精度差 | 中间计算精度不足 | 使用 FP32 中间计算 |
| UB 容量不足 | Buffer 规划不合理 | 正确计算 UB 使用量 |

---

## 六、性能优化建议

1. **Double Buffer**: 使用 `InitBuffer(que, 2, size)` 开启，CopyIn/Compute/CopyOut 并行
2. **FP16 混合精度**: Sum/Mean/Prod等计算归约场景下：BF16/FP16 输入，FP32 计算，BF16/FP16 输出
3. **A0Inner 优化**: 三约束取最小，确保负载均衡
