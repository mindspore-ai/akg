# Reduce Pattern 接口详解

跨行批量 Reduce 的 Pattern 接口高级用法。

---

## 两种重载形式

### 形式1：显式传入 sharedTmpBuffer（推荐）

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor, 
    const LocalTensor<T>& srcTensor, 
    const LocalTensor<uint8_t>& sharedTmpBuffer,  // 显式传入
    const uint32_t srcShape[], 
    bool srcInnerPad
);
```

### 形式2：框架自动申请临时空间

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor, 
    const LocalTensor<T>& srcTensor, 
    const uint32_t srcShape[], 
    bool srcInnerPad
);
```

> **⚠️ 形式2 必须预留临时空间**，否则运行时 UB 越界。详见 [临时空间预留](#临时空间预留)。

---

## Pattern 类型

| Pattern | 方向 | 输入形状 | 输出形状 | 用途 |
|---------|-----|---------|---------|------|
| `Pattern::Reduce::AR` | 沿最后一维（列方向） | (R, C) | (R,) | 每行归约为1个值 |
| `Pattern::Reduce::RA` | 沿第一维（行方向） | (R, C) | (C,) | 每列归约为1个值 |

---

## 参数说明

| 参数 | 类型 | 说明 |
|-----|------|------|
| `T` | half/float | 数据类型 |
| `pattern` | Pattern::Reduce::AR/RA | 归约模式 |
| `isReuseSource` | bool | 是否复用源操作数（默认 false） |
| `dstTensor` | LocalTensor\<T\> | 输出张量 |
| `srcTensor` | LocalTensor\<T\> | 输入张量 |
| `sharedTmpBuffer` | LocalTensor\<uint8_t\> | 临时缓存（形式1） |
| `srcShape` | uint32_t[] | `{rows, alignedCols}`，**alignedCols 必须 32 字节对齐** |
| `srcInnerPad` | bool | A2/A3 芯片只支持 `true` |

---

## 临时空间预留

**两种形式都需要预留临时空间**：

| 方式 | 预留方法 | 优点 | 推荐度 |
|-----|---------|------|-------|
| **形式1** | `InitBuffer(tmpBuf, tmpSize)` + 显式传入 | 内存可控、可复用 | ⭐⭐⭐⭐⭐ |
| **形式2** | `InitBuffer(tmpBuf, tmpSize)`（框架自动使用） | 代码简洁 | ⭐⭐⭐ |

**临时空间大小计算**：

```cpp
#include "kernel_operator.h"

uint32_t maxSize, minSize;
AscendC::GetReduceMaxMaxMinTmpSize(srcShape, sizeof(T), isReuse, maxSize, minSize);

// 使用 maxSize（安全）或 minSize（节省内存）
pipe->InitBuffer(tmpBuf, maxSize);
```

参考文档：`asc-devkit/docs/api/context/GetReduceMaxMaxMinTmpSize.md`

---

## 完整示例

### 示例1：ReduceMax（AR/RA Pattern）

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

uint32_t srcShape[] = {rows, alignedCols};  // alignedCols 必须 32 字节对齐
constexpr bool isReuse = true;

// AR Pattern：每行归约为1个值 → 输出 rows 个值
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, isReuse>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);

// RA Pattern：每列归约为1个值 → 输出 alignedCols 个值
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, isReuse>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

### 示例2：ReduceSum（框架自动申请）

```cpp
// ⚠️ 必须提前预留临时空间
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();

uint32_t srcShape[] = {rows, alignedCols};

AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
    dstLocal, srcLocal, srcShape, true);
```

---

## 对比总结

| 对比项 | 形式1（显式传入） | 形式2（框架申请） |
|-------|-----------------|------------------|
| tmp 参数 | ✅ 显式传入 | ❌ 框架自动申请 |
| 预留空间 | ✅ 调用时传入即可 | ⚠️ **必须在 InitBuffer 预留** |
| 内存管理 | 手动管理，可复用 | 需提前预留，易遗漏 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**推荐使用形式1**，避免遗漏预留空间导致运行时错误。
