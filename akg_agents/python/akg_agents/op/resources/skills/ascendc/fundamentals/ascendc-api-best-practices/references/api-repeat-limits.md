# Vector API repeatTime 参数限制

> **核心问题**：repeatTime 为 uint8_t 时最大值 255，超过会溢出导致计算错误

---

## 目录

1. [核心约束](#核心约束)
2. [如何确认参数范围](#如何确认参数范围)
3. [问题场景与解决方案](#问题场景与解决方案)
4. [受影响的 API](#受影响的-api)

---

## 核心约束

**使用 Vector API 前，必须确认 `repeatTime` 参数的范围限制！**

当 `repeatTime` 为 `uint8_t` 类型时，最大值 **255**。

---

## 如何确认参数范围

### 1. 查看函数原型中的参数类型

```cpp
// Sub API 函数原型（asc-devkit/docs/api/context/Sub.md）
template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src0,
    const LocalTensor<T>& src1,
    uint64_t mask,
    const uint8_t repeatTime,  // ← uint8_t 类型
    const BinaryRepeatParams& repeatParams
);
```

### 2. 根据数据类型确定范围

| 数据类型 | 取值范围 | 含义 |
|---------|---------|------|
| `uint8_t` | 0 ~ 255 | 最多 255 次迭代 |
| `uint16_t` | 0 ~ 65535 | 最多 65535 次迭代 |
| `int32_t` | - | 无此限制 |

---

## 问题场景与解决方案

### 问题场景

```cpp
// 问题：R=256，rowCount=256 > 255
uint32_t rowCount = 256;
AscendC::Sub<float>(
    dst[col],
    src0[col],
    src1[col],
    curMask,
    rowCount,  // uint32_t 传入 uint8_t 参数，256 溢出为 0！
    {1, 1, 1, repStride, repStride, 0}
);
```

**结果**：`rowCount=256` 被截断为 `0`，Sub 不执行任何计算，输出数据错误。

### 方案一：Host 侧限制 R_max（推荐）

```cpp
// Host 侧 R_max 计算
constexpr uint32_t MAX_REPEAT_TIMES = 255;
uint32_t R_max = (UB_SIZE - overheadBytes) / bytesPerRow;
R_max = std::min(R_max, MAX_REPEAT_TIMES);  // 确保 R_max <= 255
```

### 方案二：Kernel 侧分批处理

```cpp
void SubWithBroadcast(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    uint32_t a0Count,
    uint32_t alignedCols,
    uint32_t rowCount)
{
    constexpr uint32_t MAX_REPEAT = 255;
    uint32_t repStride = alignedCols / BLOCK_ELEMENTS;  // BLOCK_ELEMENTS = 8
    
    for (uint32_t col = 0; col < a0Count; col += MASK_FP32) {
        uint32_t curMask = std::min(a0Count - col, MASK_FP32);
        
        // 分批处理
        uint32_t processedRows = 0;
        while (processedRows < rowCount) {
            uint32_t batchRepeat = std::min(rowCount - processedRows, MAX_REPEAT);
            uint32_t rowOffset = processedRows * alignedCols;
            
            AscendC::Sub<float>(
                dst[rowOffset + col],
                src0[rowOffset + col],
                src1[col],
                curMask,
                batchRepeat,
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0}
            );
            
            processedRows += batchRepeat;
        }
    }
}
```

---

## 受影响的 API

| API 类别 | API 名称 | 参数限制 |
|---------|---------|---------|
| 二元运算 | Sub, Add, Mul, Div, Max, Min | `repeatTime` ≤ 255 |
| 一元运算 | Exp, Log, Sqrt, Abs, Neg | `repeatTime` ≤ 255 |
| 标量运算 | Muls, Adds, Divs | `repeatTime` ≤ 255 |
| 其他 | And, Or, Xor, Not | `repeatTime` ≤ 255 |

---

## 最佳实践

| 阶段 | 检查项 |
|-----|-------|
| **API 使用前** | 查看文档确认 `repeatTime` 数据类型和范围 |
| **Host Tiling** | 确保 `R_chunk_size` / `tileRows` 不超过限制 |
| **Kernel 实现** | 如需超过限制，实现分批处理逻辑 |
| **调试** | 若 R=256 时出错，优先检查 repeatTime 溢出 |

---

## 文档参考

- Sub API：`asc-devkit/docs/api/context/Sub.md`
- 其他 Vector API 文档路径相同，搜索对应 API 名称
