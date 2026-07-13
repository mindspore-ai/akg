# Reduce 类算子 - ARA Row-Split 分支

> **适用场景**: A0>1（非尾轴）, R > R_max（分载模式）

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
| **载入模式** | 分载（R 行放不下，分 chunk 处理） |
| **适用条件** | A0>1, R > R_max |
| **切分方向** | 行方向（沿 R 分 chunk） |
| **数据连续性** | 每列 R 个元素不连续（间隔 A0） |
| **Reduce 结果** | 向量（A0 个值） |

---

## 二、Buffer 规划

### 2.1 FP32 场景（Kernel侧）

```cpp
// Single Buffer 模式（分 chunk 处理）
constexpr uint32_t a0TileBase = 64;  // FP32

pipe->InitBuffer(inQueueX, 1, R_chunk_size * a0TileBase * sizeof(float));
pipe->InitBuffer(outQueueY, 1, a0TileBase * sizeof(float));  // 最终结果
pipe->InitBuffer(globalResultBuf, 1, a0TileBase * sizeof(float));  // 全局结果
pipe->InitBuffer(chunkResultBuf, 1, a0TileBase * sizeof(float));  // chunk结果
pipe->InitBuffer(tmpBuf, 1, tmpBufSize);

// 总 UB = R_chunk×a0TileBase×4 + 3×a0TileBase×4 + tmpBufSize
```

### 2.2 tmpBufSize 计算（Host侧）

```cpp
// tmpBufSize 基于 R_chunk_size × alignedCols
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

constexpr uint32_t VECTOR_REG_WIDTH = 256;
uint32_t perRepeat = VECTOR_REG_WIDTH / sizeof(float);  // 64
uint32_t perBlock = 32 / sizeof(float);  // 8

uint32_t repeats = (R_chunk_size * alignedCols + perRepeat - 1) / perRepeat;
uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * sizeof(float);
tmpBufSize = std::max(tmpBufSize, 4096u);
```

- tileA0Len是对A0进行切分分核之后，每个核需要处理的列数。
---

## 三、Tiling 参数计算

### 3.1 分载判定（Host侧）

```cpp
// R_max 计算（注意：与全载模式公式不同，此处为单缓冲，overhead 和除数均不同）
constexpr uint32_t a0TileBase = 64;  // FP32
uint32_t tmpBufSize = 4096;
uint32_t overhead = 3 * a0TileBase * sizeof(float) + tmpBufSize;
uint32_t R_max = (UB_SIZE - overhead) / (a0TileBase * sizeof(float));
R_max = std::min(R_max, 255u);
R_max = std::max(R_max, 1u);

if (R > R_max) {
    // ARA-Row-Split 模式
    loadMode = LOAD_SPLIT;
    
    // R 分 chunk
    uint32_t R_chunks = (R + R_max - 1) / R_max;
    uint32_t R_chunk_size = R_max;
    uint32_t R_last_chunk_size = R - (R_chunks - 1) * R_chunk_size;
}
```

### 3.2 A0Inner 计算（对A0进行多核切分，Host侧）

```cpp
// Row-Split buffer 使用: R_chunk×a0TileBase×4 + 3×a0TileBase×4 + tmpBufSize
uint64_t ubPerTileBase = R_chunk_size * a0TileBase * sizeof(float) 
                       + 3 * a0TileBase * sizeof(float) + tmpBufSize;
uint64_t fixedOverhead = 3 * a0TileBase * sizeof(float) + tmpBufSize;

int64_t factorMax = (UB_SIZE - fixedOverhead) / ubPerTileBase;
if (factorMax < 1) factorMax = 1;

int64_t a0FactorMax = (A0 + a0TileBase - 1) / a0TileBase;
int64_t totalTilesMax = A1 * a0FactorMax;
int64_t a0InnerMax = (totalTilesMax + blockDim - 1) / blockDim;

int64_t a0Inner = std::min({a0InnerMax, factorMax, a0FactorMax});
a0Inner = std::max(a0Inner, 1L);

uint32_t tileA0Len = a0Inner * a0TileBase;
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM (A1, R, A0) → 分 R chunk 处理
    ↓
Chunk 0: GM[0:R_chunk, 0:tileA0Len] → UB
    ↓
[ReduceMax/ReduceSum Pattern::Reduce::RA] → chunkResult_0 (tileA0Len)
    ↓
[更新 globalResult] → globalResult = merge(globalResult, chunkResult_0)
    ↓
Chunk 1, 2, ... (重复)
    ↓
UB → GM (A1, A0)
```

**为什么选择 Pattern::Reduce::RA**:

多核切分后，每个核处理部分 `(R_chunk, A0_inner)` 的数据，在 UB 中布局为：
```
[row0的全部A0_inner, row1的全部A0_inner, ..., row{R_chunk-1}的全部A0_inner]
```
即 `(R_chunk, alignedCols)` 的 2D 矩阵。

Pattern::Reduce::RA 的含义：
- **R** = Reduce 维度（沿第一维归约）
- **A** = Align 维度（保留第二维）

对于每个 `a0` 位置，取 R_chunk 个值归约，输出 A0_inner 个结果。

**注意**：不能用 Level 2 API，因为它只能处理连续数据。

### 4.2 核心 API 调用

#### ReduceMax 分载实现

```cpp
// 对齐后的列数
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// 初始化全局最大值为 -∞
LocalTensor<float> globalResultLocal = globalResultBuf.Get<float>();
Duplicate<float>(globalResultLocal, -INFINITY, alignedCols);

for (uint32_t chunkIdx = 0; chunkIdx < R_chunks; chunkIdx++) {
    uint32_t rStart = chunkIdx * R_chunk_size;
    uint32_t rCount = (chunkIdx == R_chunks - 1) ? R_last_chunk_size : R_chunk_size;
    
    // Load R chunk
    DataCopyExtParams copyParams{static_cast<uint16_t>(rCount), 
                                  static_cast<uint32_t>(tileA0Len * sizeof(float)), 
                                  static_cast<uint32_t>((A0 - tileA0Len) * sizeof(float)), 
                                  0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    // shape will be [rCount, alignedCols] with `[rCount, tileA0Len]` valid data.
    DataCopyPad(xLocal, xGm[rStart * A0], copyParams, padParams);
    
    // ReduceMax for this chunk (Pattern::Reduce::RA)
    uint32_t srcShape[] = {rCount, alignedCols};
    LocalTensor<float> chunkResultLocal = chunkResultBuf.Get<float>();
    ReduceMax<float, Pattern::Reduce::RA>(chunkResultLocal, xLocal, tmpLocal, srcShape, true);
    
    // Update globalResult (逐元素取最大值)
    Max<float>(globalResultLocal, globalResultLocal, chunkResultLocal, alignedCols);
}

// 输出最终结果
DataCopyPad(yGm[offset], globalResultLocal, {1, tileA0Len * sizeof(float), 0, 0});
```

### 4.3 关键注意点

1. **跨 chunk 合并**: 
   - ReduceMax: 使用 `Max<float>` 逐元素取最大值
   - ReduceSum: 使用 `Add<float>` 逐元素累加

2. **边界处理**: 最后一个 R chunk 的大小可能小于 R_chunk_size

3. **初始化**: 
   - ReduceMax: 初始化为 `-INFINITY` 或数据类型的最小值
   - ReduceSum: 初始化为 `0`

4. **数据访问**: 需要正确计算 srcStride 跨越 A0

---

## 五、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 精度下降 | chunk 合并逻辑错误 | 使用 Max/Add 逐元素合并 |
| 输出错误 | 最后一个 R chunk 大小处理错误 | 使用 R_last_chunk_size |
| 性能差 | 多次遍历数据 | 优化 chunk 策略 |
| Buffer 不足 | Buffer 规划不合理 | 合理分配 GM/UB 空间 |
| 精度过大 | srcShape 使用 A0 而非 alignedCols | 使用对齐后的列数 |

---

## 六、性能优化建议

1. **减少数据访问**: 尽量减少 GM 访问次数
2. **chunk 大小优化**: R_chunk_size = R_max，确保每个 chunk 充分利用 UB
3. **流水线**: 不同 tile 的 chunk 可以并行处理
4. **逐元素合并**: 使用 `Max<float>` 和 `Add<float>` 而非 Reduce 操作
