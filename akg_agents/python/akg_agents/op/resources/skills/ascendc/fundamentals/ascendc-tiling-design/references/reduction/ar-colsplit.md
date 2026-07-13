# Reduce 类算子 - AR Col-Split 分支

> **适用场景**: A0=1（尾轴）, R > threshold（分载模式）

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
| **模板类型** | AR 模板（A 轴 + R 轴） |
| **Shape 抽象** | (A1, R) |
| **载入模式** | 分载（整行放不下，分 chunk 处理） |
| **适用条件** | A0=1, R > [全载阈值](#3.1全载阈值) |
| **切分方向** | 列方向（沿 R 分 chunk） |
| **数据连续性** | 每行 R 个元素连续 |
| **Reduce 结果** | 标量（1 个值） |

---

## 二、Buffer 规划

### 2.1 FP32 场景（Kernel侧）

```cpp
// Single Buffer 模式（分 chunk 处理）
pipe->InitBuffer(inQueueX, 1, chunkCols * sizeof(float));
pipe->InitBuffer(outQueueY, 1, 32);  // Reduce结果（1个标量）
pipe->InitBuffer(chunkResultBuf, 1, 32);  // chunk中间结果
pipe->InitBuffer(tmpBuf, 1, tmpBufSize);

// 总 UB = chunkCols×4 + 32 + 32 + tmpBufSize
```

### 2.2 chunkCols 计算（Host侧）

```cpp
// 基于 UB 容量计算：完成1行数据计算时，支持的最大列数chunkCols
chunkCols = std::min(chunkCols, R);
uint32_t numChunks = (R + chunkCols - 1) / chunkCols;
uint32_t lastChunkSize = R - (numChunks - 1) * chunkCols;
```

---

## 三、Tiling 参数计算

### 3.1 分载判定

解释：在UB中完成1行数据计算时，支持的最大列数chunkCols。若R > chunkCols，则需要分载。

### 3.2 多核切分参数（Host侧）

```cpp
// 按 A1（行）切分
uint32_t rowsPerCore = (A1 + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 % rowsPerCore;
if (tailCoreRows == 0 && A1 > 0) tailCoreRows = rowsPerCore;
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM (A1, R) → 分 chunk 处理
    ↓
Chunk 0: GM[0:chunkCols] → UB
    ↓
[ReduceMax] → chunkResult_0
    ↓
[更新 globalResult] → globalResult = merge(globalResult, chunkResult_0)
    ↓
Chunk 1, 2, ... (重复)
    ↓
UB → GM (A1)
```

### 4.2 核心 API 调用

#### ReduceMax 分载实现（Kernel侧）

```cpp
// 初始化全局最大值为 -∞
float globalMax = -INFINITY;

for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    uint32_t chunkStart = chunkIdx * chunkCols;
    uint32_t chunkSize = (chunkIdx == numChunks - 1) ? lastChunkSize : chunkCols;
    
    // Load chunk. use `DataCopyPad` in case last chunk is not 32 bytes aligned.
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(chunkSize * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[chunkStart], copyParams, padParams);
    
    // ReduceMax for this chunk
    ReduceMax<float>(chunkResultLocal, xLocal, tmpLocal, chunkSize, false);
    float chunkMax = chunkResultLocal.GetValue(0);
    
    // Update `globalMax`. DO NOT use `std::` function.
    if (chunkMax > globalMax) {
        globalMax = chunkMax;
    }
}

// use `globalMax` to complete the reset calculation...
```

### 4.3 关键注意点

1. **跨 chunk 合并**: 
   - ReduceMax: 使用 `if` 或者 `三目运算` 取最大值
   - ReduceSum: 使用 `+=` API 累加
2. **边界处理**: 最后一个 chunk 的大小可能小于 chunkCols。搬入/搬出时需要使用`CopyDataPad`
3. **初始化**: 
   - ReduceMax: 初始化为 `-INFINITY` 或数据类型的最小值
   - ReduceSum: 初始化为 `0`

---

## 五、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 精度下降 | chunk 合并逻辑错误 | 确保正确合并（Max取最大值，Sum累加） |
| 输出错误 | 最后一个 chunk 大小处理错误 | 使用 lastChunkSize 而非 chunkCols |
| 性能差 | 多次遍历数据 | 优化 chunk 策略，减少遍历次数 |
| Buffer 不足 | chunkCols 计算错误 | 基于 UB 容量正确计算 |

---

## 六、性能优化建议

1. **chunk 大小优化**: 根据 UB 容量选择最优 chunk 大小，尽量大以减少chunk数量
2. **避免不必要的数据拷贝**: 直接在chunk结果上合并，减少中间存储
