# Reduce 类算子 - AR 全载分支

> **适用场景**: A0=1（尾轴）, R ≤ threshold（全载模式）

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
| **载入模式** | 全载（整行完整放入 UB） |
| **适用条件** | A0=1, R ≤ [全载阈值](#3.1全载阈值) |
| **数据连续性** | 每行 R 个元素连续 |
| **Reduce 结果** | 标量（1 个值） |

---

## 二、Buffer 规划

### 2.1 FP32 场景（Kernel侧）

```cpp
// Double Buffer 模式
pipe->InitBuffer(inQueueX, 2, R * sizeof(float));
pipe->InitBuffer(outQueueY, 2, 32);  // Reduce结果（1个标量）
pipe->InitBuffer(tmpBuf, tmpBufSize);

// 总 UB = 2×R×4 + 2×32 + tmpBufSize
```

### 2.2 tmpBufSize 计算（Host侧）

```cpp
uint32_t ComputeReduceBufSize(uint32_t rLengthAlign, uint32_t typeSize) {
    uint32_t perRepeat = 256 / typeSize;  // 64 for FP32
    uint32_t perBlock = 32 / typeSize;     // 8 for FP32
    uint32_t repeats = (rLengthAlign + perRepeat - 1) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * typeSize;
    return std::max(tmpBufSize, 4096u);  // 最小 4KB
}
```

---

## 三、Tiling 参数计算

### 3.1 全载阈值

解释：在UB中完成1行数据计算时，支持的最大列数R_max。

### 3.2 多核切分参数（Host侧）

```cpp
// 按 A1（行）切分
uint32_t rowsPerCore = (A1 + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 % rowsPerCore;
if (tailCoreRows == 0 && A1 > 0) tailCoreRows = rowsPerCore;
```

### 3.3 对齐处理（Kernel侧）

```cpp
// 计算对齐后的列数
uint32_t alignedCols = ((R * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// 使用 DataCopyPad 处理非对齐（只拷贝一行。若UB较富余，可以多行批量拷贝）
DataCopyExtParams copyParams{1, static_cast<uint32_t>(R * sizeof(float)), 0, 0, 0};
DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM (A1, R) → UB (R)
    ↓
[ReduceMax/ReduceSum] → result (1)
    ↓
UB (1) → GM (A1)
```

### 4.1 核心 API 调用（Kernel侧）

> **推荐**：AR 全载使用 **Level 2 接口（逐行处理）**，更简单且无对齐要求

```cpp
for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;  // ⚠️ 关键：用对齐后的长度
    
    // ReduceMax - 使用有效数据个数
    ReduceMax<T>(resultLocal, xLocal[rowOffset], tmpLocal, 
                 static_cast<int32_t>(rLength), false);
    
    // 或 ReduceSum
    ReduceSum<T>(resultLocal, xLocal[rowOffset], tmpLocal, 
                 static_cast<int32_t>(rLength), false);
    
    // 输出结果
    T result = resultLocal.GetValue(0);
    // ... 后续处理
}
```

**关键要点**:
- `rowOffset` 计算：用 `rLengthAlign`（UB 中每行按 32 字节对齐存储）
- API `count` 参数：用 `rLength`（只处理有效数据，不包括 padding）
- Buffer 大小：用 `rLengthAlign`（需要容纳对齐后的数据）

### 4.2 流水线设计

**Double Buffer 模式** (`InitBuffer(que, 2, size)`):
```
Tile N:   CopyIn(row0) → Compute(row0) → CopyOut(row0)
Tile N+1:              CopyIn(row1) → Compute(row1) → CopyOut(row1)
```

---

## 五、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 输出全 0 | Buffer 未正确初始化 | 检查 AllocTensor/FreeTensor 配对 |
| FP16 精度差 | 中间计算精度不足 | 使用 FP32 中间计算 |
| 非对齐场景精度错误 | count 用 rLengthAlign 而非 rLength | count 参数用 `rLength`（有效数据个数） |
| 多行数据错误 | rowOffset 计算错误 | rowOffset 用 `rLengthAlign`（UB 按对齐存储） |

---

## 六、性能/精度优化建议

1. **Double Buffer**: 使用 `InitBuffer(que, 2, size)` 开启, CopyIn/Compute/CopyOut 并行
2. **FP16 混合精度**: Sum/Mean/Prod等计算归约场景下：BF16/FP16 输入，FP32 计算，BF16/FP16 输出
