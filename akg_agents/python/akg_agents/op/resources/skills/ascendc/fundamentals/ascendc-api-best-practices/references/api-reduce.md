# Reduce API 使用指南

逐行 Reduce 与跨行 Reduce 的 API 选择与使用规范。

---

## 目录

1. [接口选择](#接口选择)
2. [Level 2 接口（逐行处理）](#level-2-接口逐行处理)
3. [Pattern 接口（跨行批量）](#pattern-接口跨行批量)
4. [常见错误](#常见错误)
5. [最佳实践](#最佳实践)

---

## 接口选择

| 场景 | 接口 | 参数 | 对齐要求 | 典型用途 |
|-----|------|------|----------|---------|
| 逐行独立处理 | Level 2 | `(dst, src, tmp, count)` | **无** | Softmax, LayerNorm |
| 跨行批量处理 | Pattern | **两种形式**（见下文） | 32 字节 | ReduceSum axis=-1 |

**选择原则**：
- 逐行独立计算 → **Level 2 接口**（更简单，无对齐要求）
- 需要跨行 Reduce → **Pattern 接口**（性能更高，推荐形式1）

---

## Level 2 接口（逐行处理）

### API 签名

```cpp
AscendC::ReduceMax<T>(dst, src, tmpBuffer, count, calIndex);
AscendC::ReduceSum<T, isSetMask=true>(dst, src, tmpBuffer, count);
AscendC::ReduceMin<T>(dst, src, tmpBuffer, count, calIndex);
```

**参数**：
- `dst`：输出 LocalTensor（1 个元素）
- `src`：输入 LocalTensor（count 个元素）
- `tmpBuffer`：临时 buffer（**类型必须与 T 相同**）
- `count`：元素个数（int32_t）
- `calIndex`：是否计算索引（bool，默认 false）

### tmpBuffer 类型要求

**tmpBuffer 类型必须与 dst/src 相同**：

```cpp
// ❌ 错误：tmpBuffer 类型不匹配
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);  // 编译错误！

// ✅ 正确：tmpBuffer 类型必须与 T 相同
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

### 完整示例：Softmax 逐行处理

```cpp
__aicore__ inline void ProcessRow(
    AscendC::LocalTensor<T>& xLocal, 
    AscendC::LocalTensor<T>& yLocal, 
    uint32_t rowIdx)
{
    uint32_t rowOffset = rowIdx * rLengthAlign;  // ⚠️ 用 rLengthAlign
    
    AscendC::LocalTensor<T> rowTmp = rowBuf.Get<T>();
    AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
    
    // 1. ReduceMax（count = rLength，有效数据个数）
    AscendC::ReduceMax<T>(rowTmp, xLocal[rowOffset], reduceTmp, 
        static_cast<int32_t>(rLength), false);
    
    T maxVal = rowTmp.GetValue(0);
    AscendC::Duplicate<T>(rowTmp, maxVal, rLength);
    AscendC::Sub<T>(xLocal[rowOffset], xLocal[rowOffset], rowTmp, rLength);
    
    // 2. Exp
    AscendC::Exp<T>(xLocal[rowOffset], xLocal[rowOffset], rLength);
    
    // 3. ReduceSum
    AscendC::ReduceSum<T, true>(rowTmp, xLocal[rowOffset], reduceTmp, 
        static_cast<int32_t>(rLength));
    
    T sumVal = rowTmp.GetValue(0);
    AscendC::Duplicate<T>(rowTmp, sumVal, rLength);
    AscendC::Div<T>(yLocal[rowOffset], xLocal[rowOffset], rowTmp, rLength);
}
```

---

## Pattern 接口（跨行批量）

Pattern 接口有**两种重载形式**，详见 [api-reduce-pattern.md](api-reduce-pattern.md)。

### 快速入门

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

uint32_t srcShape[] = {rows, alignedCols};  // alignedCols 必须 32 字节对齐

// 推荐使用形式1：显式传入 tmpLocal
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

### 关键要点

| 要点 | 说明 |
|-----|------|
| **对齐要求** | `alignedCols` 必须 32 字节对齐 |
| **Pattern 类型** | `Pattern::Reduce::AR`（沿列方向）、`Pattern::Reduce::RA`（沿行方向） |
| **推荐形式** | 形式1（显式传入 sharedTmpBuffer） |
| **临时空间** | 两种形式都需要预留，详见 [api-reduce-pattern.md](api-reduce-pattern.md) |

### 非对齐数据处理

```cpp
// ✅ 方案1：改用 Level 2 接口（无对齐要求）
AscendC::ReduceMax<T>(dst, src, tmp, rLength, false);

// ✅ 方案2：用 DataCopyPad 填充到对齐
uint32_t alignedCols = ((rLength * sizeof(T) + 31) / 32) * 32 / sizeof(T);
AscendC::DataCopyPadExtParams<T> padParams;
padParams.isPad = true;
padParams.rightPadding = alignedCols - rLength;
DataCopyPad(dstLocal, srcGm, copyParams, padParams);

uint32_t srcShape[] = {1, alignedCols};
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);
```

---

## 常见错误

### 错误1：tmpBuffer 类型不匹配

```cpp
// ❌ 错误
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);

// ✅ 正确
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

### 错误2：rowOffset 用 rLength 而非 rLengthAlign

```cpp
// ❌ 错误：单行通过，多行失败
uint32_t rowOffset = rowIdx * rLength;

// ✅ 正确
uint32_t rowOffset = rowIdx * rLengthAlign;
```

### 错误3：非对齐数据用 Pattern 接口

```cpp
// ❌ 错误：rLength=13，非 32 字节对齐
uint32_t srcShape[] = {1, rLength};
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);

// ✅ 方案1：改用 Level 2 接口
AscendC::ReduceMax<T>(dst, src, tmp, rLength, false);

// ✅ 方案2：用 DataCopyPad 填充到对齐（见上文）
```

### 错误4：Reduce API count 传 rLengthAlign

```cpp
// ❌ 错误：count 应该是有效数据个数
AscendC::ReduceMax(rowTmp, src, tmp, rLengthAlign, false);

// ✅ 正确：count 只传有效数据个数
AscendC::ReduceMax(rowTmp, src, tmp, rLength, false);
```

### 错误5：Pattern 接口形式2 忘记预留临时空间

```cpp
// ❌ 错误：运行时 UB 越界或结果错误
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);

// ✅ 方案1：使用形式1（推荐）
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(dst, src, tmpLocal, srcShape, true);

// ✅ 方案2：预留临时空间（详见 api-reduce-pattern.md）
```

### 错误6：Reduce dst 起始地址未 8 字节对齐

`ReduceMax<float>` / `ReduceSum<float>` 等 Reduce API 要求 **dst 起始地址 8 字节对齐**（对 fp32 即 2 个元素对齐）。在"小组归约"场景下（每行多组、每组结果仅占 4 字节）容易出现奇数 offset 位置违反对齐。

```cpp
// ❌ dst 用 stride 1 fp32：每组结果占 4 字节，
// 写到 dstBuf[r * groupsPerRow + g] 在 g 为奇数时只满足 4B 对齐
const uint32_t groupsPerRow = 4;
AscendC::ReduceMax<float>(dstBuf[r * groupsPerRow + g], src, tmp, 32, false);  // g=1,3 → 4B 对齐
```

修复：dst buffer 用 stride 2 fp32（每组结果占 8 字节槽位）：

```cpp
// ✅ stride 2 fp32 → 每个 dst 结果占 8 字节，任何 g 都满足 8B 对齐
AscendC::ReduceMax<float>(dstBuf[r * groupsPerRow * 2 + g * 2], src, tmp, 32, false);
```

下游读取时索引同步乘 2。

**症状**：Reduce API 返回静默错误（结果残留旧值或写到错位置），不一定立刻 trap。

---

## 最佳实践

### 参数对照表

| 参数位置 | 用 rLength | 用 rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer 大小计算 | ✗ | ✓ |

### 决策流程

```
需要 Reduce 操作？
    │
    ├─ 逐行独立处理（Softmax/LayerNorm）
    │     └─→ Level 2 接口
    │           - 无对齐要求
    │           - count = rLength
    │
    └─ 跨行批量 Reduce
          └─→ Pattern 接口（形式1 推荐）
                - 需要 32 字节对齐
                - 显式管理 tmp buffer
```

### Buffer 分配

```cpp
uint32_t tileSize = rowsPerLoop * rLengthAlign * sizeof(T);
uint32_t rowBufSize = rLengthAlign * sizeof(T);
uint32_t reduceBufSize = 32 * 1024;

pipe->InitBuffer(inQueueX, 1, tileSize);
pipe->InitBuffer(outQueueY, 1, tileSize);
pipe->InitBuffer(rowBuf, rowBufSize);
pipe->InitBuffer(reduceBuf, reduceBufSize);
```

---

## API 文档查阅优先级

1. ⭐⭐⭐ **官方 API 文档**：`asc-devkit/docs/api/context/ReduceMax.md`
2. ⭐⭐⭐ **官方示例代码**：`asc-devkit/examples/03_libraries/05_reduce/`
3. Pattern 接口详解：[api-reduce-pattern.md](api-reduce-pattern.md)

---

## 参考示例

- `asc-devkit/examples/03_libraries/05_reduce/reducemax/reducemax.asc` - Pattern 接口示例
- `asc-devkit/docs/api/context/ReduceMax.md` - 官方 API 文档
