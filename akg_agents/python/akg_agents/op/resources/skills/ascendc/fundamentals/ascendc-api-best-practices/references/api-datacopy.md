# DataCopy / DataCopyPad 使用指南

GM ↔ UB 数据搬运的完整指南。

---

## 目录

1. [选择规则](#选择规则)
2. [32 字节对齐要求](#32-字节对齐要求)
3. [DataCopyPad 参数详解](#datacopypad-参数详解)
4. [使用场景示例](#使用场景示例)
5. [stride 参数详解](#stride-参数详解)
6. [常见错误与调试](#常见错误与调试)

---

## 选择规则

**原则：优先使用 DataCopyPad**

| 场景 | API | 原因 |
|-----|-----|------|
| **非对齐或不确定对齐** | `DataCopyPad` | 自动处理对齐/非对齐，避免边界 bug |
| **数据量严格 32 字节对齐** | `DataCopy` 或 `DataCopyPad` | 确定对齐时 DataCopy 可用，DataCopyPad 更安全 |

### ⛔️ 黑名单 API（禁止在生产代码中使用）

| API | 禁止原因 | 仅允许场景 |
|-----|---------|-----------|
| `GlobalTensor::SetValue(idx, val)` | 效率极低，单元素逐个写入 | **仅调试时使用** |
| `GlobalTensor::GetValue(idx)` | 效率极低，单元素逐个读取 | **仅调试时使用** |

```cpp
// ❌ 禁止：生产代码使用 SetValue/GetValue
for (uint32_t i = 0; i < size; i++) {
    xGm.SetValue(i, value);    // ⛔️ 效率极低
    T val = xGm.GetValue(i);   // ⛔️ 效率极低
}

// ✅ 正确：使用 DataCopyPad 批量搬运
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// ✅ 允许：调试时单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));  // 仅调试
```

**为什么优先 DataCopyPad？**

1. 自动处理非对齐，无需手动判断
2. CopyIn 和 CopyOut 都适用
3. Tiling 设计时可能产生非对齐的 tile 大小
4. 对齐场景下性能差异可忽略

---

## 32 字节对齐要求

**DataCopy 要求 32 字节对齐**，非对齐会导致数据错误。

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |
| fp8 (1 byte) | 32 | 32 |

---

## DataCopyPad 参数详解

### isPad 参数

| isPad | 含义 |
|-------|------|
| `false` | 框架自动填充，用户不指定填充值 |
| `true` | 用用户指定的 `paddingValue` 填充 |

### blockLen 非对齐时的填充行为

**GM → UB（CopyIn）**：

| 条件 | isPad | dummy 填充值 |
|-----|-------|-------------|
| leftPadding=0, rightPadding=0 | false | **第一个元素值** |
| leftPadding=0, rightPadding=0 | true | paddingValue |
| leftPadding≠0 或 rightPadding≠0 | false | 随机值 |
| leftPadding≠0 或 rightPadding≠0 | true | paddingValue |

**UB → GM（CopyOut）**：
- 框架自动处理非对齐
- 搬到 GM 时自动丢弃 dummy

### UB 端起始地址 32B 对齐（易踩坑）

`DataCopyPad(GM, UB, ...)` 与 `DataCopyPad(UB, GM, ...)` 的 **UB 端起始地址必须 32 字节对齐**（blockLen 可以非 32B 对齐，但起始地址不能）。

按行索引访问 UB buffer 时，行偏移字节数必须是 32 的倍数：

```cpp
// ❌ cols * sizeof(elem) 不是 32 倍数时，row * cols 偏移可能落非对齐地址
// 例如 fp8 + cols=4 时每行 4 字节，只有 row ∈ {0, 8, 16,...} 满足 32B 对齐
DataCopyPad(gmOut[off], ubBuf[row * cols], copyParams);
```

修复方式：引入 strided staging buffer，把不规则行宽数据重排到每行 32B 对齐的连续区：

```cpp
// ✅ 用 strided buf 重排，保证每行 UB src 起址 32B 对齐
auto stridedBuf = strideBuf_.Get<elem_T>();
for (int row = 0; row < mEff; ++row) {
    for (int j = 0; j < cols; ++j) {
        stridedBuf.SetValue(row * 32 + j, ubBuf.GetValue(row * cols + j));
    }
}
DataCopyPad(gmOut[off], stridedBuf[row * 32], copyParams);  // src 每行 32B 对齐
```

**错误码症状**：
- `AIV error 80: The UB address accessed by the VEC instruction is not aligned`
- 连锁触发 `AIC error: timeout or trap error. subErrType: 0x4`

### blockCount 参数限制

`DataCopyPad` 的 `blockCount` 字段最大值 4095，超出需分批搬运。Host 侧 Tiling 计算必须 clip：

```cpp
constexpr uint32_t MAX_BLOCK_COUNT = 4095;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

---

## 使用场景示例

### 场景1：非对齐 CopyIn，不关心填充值

```cpp
// cols=5 (FP32)，blockLen=20字节，非对齐
// 后续计算只处理 cols 个元素，dummy 被忽略
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// 后续计算只处理 cols 个元素
AscendC::ReduceMax(tmpReduce, xLocal, tmpReduce, cols, false);
```

### 场景2：非对齐 CopyIn，指定填充值

```cpp
uint32_t padElements = paddedCols - cols;
AscendC::DataCopyPadExtParams<float> padParams{true, 0, padElements, 0.0f};
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 场景3：非对齐 CopyOut

```cpp
// CopyOut 自动处理非对齐，搬到 GM 时丢弃 dummy
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 完整示例：多行批量搬运

```cpp
__aicore__ inline void CopyInBatch(uint32_t startLocalRow, uint32_t rowsThisTile)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = rowsThisTile;
    copyParams.blockLen = cols * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    
    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = paddedColsT - cols;
    padParams.paddingValue = 0;
    
    AscendC::DataCopyPad(xLocal, xGm[startLocalRow * cols], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}
```

### 场景4：逐行独立处理模式（Softmax/LayerNorm 推荐）

**适用场景**：Softmax / LayerNorm 等逐行独立计算的算子，不需要跨行 Reduce。

**核心要点**：blockCount 模式 + UB 对齐存储

```cpp
// ========== Tiling 参数 ==========
uint32_t rLength = 13;                           // 有效数据个数
uint32_t rLengthAlign = (rLength + 7) / 8 * 8;   // 对齐到 8 元素（FP32 下 32 字节）

// ========== 数据搬运 ==========
AscendC::DataCopyPad(xLocal, xGm[offset], 
    {static_cast<uint16_t>(rows),           // blockCount: 行数
     static_cast<uint32_t>(rLength * sizeof(T)), // blockLen: 有效数据长度（非对齐！）
     0, 0},                                  // stride: 连续存储
    {false, 0, 0, 0});                       // padParams: 自动处理

inQueueX.EnQue(xLocal);
auto xIn = inQueueX.DeQue<T>();

// ========== 逐行处理 ==========
for (uint32_t row = 0; row < rows; row++) {
    // 关键：UB 偏移用 rLengthAlign，不是 rLength！
    uint32_t rowOffset = row * rLengthAlign;
    
    // Reduce API 只传 rLength（有效数据个数）
    AscendC::ReduceMax<T>(rowTmp, xIn[rowOffset], reduceTmp, 
        static_cast<int32_t>(rLength), false);
    // ... Sub, Exp, ReduceSum, Div
}

// ========== 写回 GM ==========
AscendC::DataCopyPad(yGm[offset], yOut, 
    {static_cast<uint16_t>(rows), static_cast<uint32_t>(rLength * sizeof(T)), 0, 0, 0});
```

**关键对照表**：

| 参数位置 | 用 rLength | 用 rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer 大小 | ✗ | ✓ |

**UB 数据布局示意**：

```
GM（连续存储）:  [row0: 13元素][row1: 13元素][row2: 13元素]...
                         ↓ DataCopyPad blockCount 模式
UB（对齐存储）:  [row0: 13+3=16][row1: 13+3=16][row2: 13+3=16]...
                         ↑
                  每行 padding 到 8 元素对齐
```

---

## stride 参数详解

**stride 参数单位取决于操作数位置**：

| 操作数位置 | stride 单位 | 说明 |
|-----------|------------|------|
| GlobalTensor (GM) | **字节** | 相邻数据块的字节间隔 |
| LocalTensor (UB) | **dataBlock (32字节)** | 相邻数据块的 32字节块间隔 |

**stride 含义**：相邻数据块之间的间隔（前一块尾部到后一块头部的距离）

### UB → GM 多行搬运（CopyOut）

```cpp
// UB 中每行: [cols 有效数据][padElements padding]
// 相邻行间隔 = paddedColsT - cols 个元素
copyParams.blockCount = rowsThisTile;
copyParams.blockLen = cols * sizeof(T);
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;  // UB stride 单位: 32字节
copyParams.dstStride = 0;  // GM stride 单位: 字节

AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 常见错误

```cpp
// ❌ 错误：srcStride 理解为行长度
copyParams.srcStride = paddedColsT * sizeof(T) / 32;  // 这会导致输出错位

// ✅ 正确：srcStride 是间隔
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;
```

---

## 常见错误与调试

### 错误1：CopyIn/CopyOut 非对齐数据用 DataCopy

```cpp
// ❌ 错误
AscendC::DataCopy(xLocal, xGm, 4);  // cols=4 (16 bytes)，数据错误

// ✅ 正确
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 错误2：CopyIn 用 DataCopyPad，CopyOut 用 DataCopy

```cpp
// ❌ 错误：CopyIn 和 CopyOut 都需要处理非对齐
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopy(yGm, yLocal, 4);  // 输出错误

// ✅ 正确：两边都用 DataCopyPad
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 调试步骤

遇到数据错误时：

1. **分别验证 CopyIn 和 CopyOut**
   - 用 "CopyIn → CopyOut" 测试搬运是否正确
2. **检查数据量是否 32 字节对齐**
3. **非对齐场景：CopyIn 和 CopyOut 都用 DataCopyPad**

### 实战案例：SoftmaxV5

**问题**：FP32 cols=4,5,6,7 时结果错误，cols=8 正常

**根因**：
1. CopyIn 用 DataCopyPad 但 isPad=false（填充随机值）
2. CopyOut 用 DataCopy 处理非对齐输出

**解决**：CopyIn 和 CopyOut 都用 DataCopyPad
