# 算术运算 API 优化指南

> **适用场景**：使用算术运算 API（Add/Sub/Mul/Div）时，选择最优实现方式，避免不必要的广播 buffer 和指令开销。

---

## 目录

- [概述](#概述)
- [场景1：标量操作（单行）](#场景1标量操作单行)
  - [方案对比](#方案对比)
  - [API 接口](#api-接口)
  - [完整示例](#完整示例)
- [场景2：广播操作（多行）](#场景2广播操作多行)
  - [方案对比-1](#方案对比-1)
  - [核心原理](#核心原理)
  - [分批处理](#分批处理)
- [场景3：半精度加减法精度优化](#场景3半精度加减法精度优化)
  - [问题根因](#问题根因)
  - [默认策略](#默认策略)
  - [标准范式](#标准范式)
  - [Kernel 集成要点](#kernel-集成要点)
- [性能对比](#性能对比)
- [适用 API](#适用-api)
- [常见错误](#常见错误)

---

## 概述

算术运算 API（Add/Sub/Mul/Div）支持两种使用模式：

| 模式 | API | 适用场景 | Buffer 需求 |
|-----|-----|---------|------------|
| **标量操作** | `Adds/Muls` | 单行处理（Softmax AR 模板） | 32B |
| **广播操作** | `Sub/Div + BinaryRepeatParams` | 多行处理（Softmax ARA 模板） | alignedCols×4 |

**关键优化**：
- 单行：使用 `Adds/Muls` 避免 Duplicate
- 多行：使用 `src1RepStride=0` 避免逐行循环

---

## 场景1：标量操作（单行）

### 方案对比

**问题**：需要对 tensor 每个元素执行 `x - scalar` 或 `x / scalar`

**典型场景**：
- Softmax AR 模板：`x - max_val`（数值稳定）
- Softmax AR 模板：`exp(x) / sum`（归一化）
- LayerNorm：`x - mean`（中心化）
- BatchNorm：`x * gamma + beta`

**方案对比**：

| 方案 | 指令数 | Buffer 需求 | 推荐度 |
|-----|--------|------------|--------|
| Duplicate + Sub | 2 条 | `rLength × sizeof(T)` | ⭐⭐ |
| Duplicate + Div | 2 条 | `rLength × sizeof(T)` | ⭐⭐ |
| **Adds(-scalar)** | **1 条** | **32B** | **⭐⭐⭐⭐⭐** |
| **Muls(1/scalar)** | **1 条** | **32B** | **⭐⭐⭐⭐⭐** |

### API 接口

**Adds（标量加法）**：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(
    const LocalTensor<T>& dst, 
    const LocalTensor<T>& src, 
    const T& scalarValue, 
    const int32_t& count);

// 功能: dst[i] = src[i] + scalarValue
// 示例: Adds(dst, src, -maxVal, count)  // 减法转加法
```

**Muls（标量乘法）**：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(
    const LocalTensor<T>& dst, 
    const LocalTensor<T>& src, 
    const T& scalarValue, 
    const int32_t& count);

// 功能: dst[i] = src[i] * scalarValue
// 示例: Muls(dst, src, 1.0/sum, count)  // 除法转乘法
```

### 完整示例

#### 优化前（Sub/Div + Duplicate）

```cpp
// Buffer 初始化
uint32_t broadcastBufSize = rLengthAlign * sizeof(T);  // 例如：512B (rLength=128, FP32)
pipe.InitBuffer(broadcastBuf, broadcastBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> broadcastLocal = broadcastBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;
    
    // Step 1: ReduceMax
    ReduceMax<T>(broadcastLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);
    
    // Step 2: Duplicate + Sub（需要广播 buffer）
    T maxVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, maxVal, rLength);  // 指令 1
    Sub<T>(yLocal[rowOffset], xLocal[rowOffset], broadcastLocal, rLength);  // 指令 2
    
    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);
    
    // Step 4: ReduceSum
    ReduceSum<T, true>(broadcastLocal, yLocal[rowOffset], reduceTmpLocal, rLength);
    
    // Step 5: Duplicate + Div（需要广播 buffer）
    T sumVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, sumVal, rLength);  // 指令 3
    Div<T>(yLocal[rowOffset], yLocal[rowOffset], broadcastLocal, rLength);  // 指令 4
}

// 总计：6 条指令/行，需要 broadcastBuf (512B for rLength=128)
```

#### 优化后（Adds/Muls + 标量）

```cpp
// Buffer 初始化（节省 broadcastBuf）
uint32_t scalarBufSize = 32;  // 最小对齐要求，仅需存储 1 个标量
pipe.InitBuffer(scalarBuf, scalarBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> scalarLocal = scalarBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;
    
    // Step 1: ReduceMax
    ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);
    
    // Step 2: Adds（直接标量操作，无需广播）
    T maxVal = scalarLocal.GetValue(0);
    Adds<T>(yLocal[rowOffset], xLocal[rowOffset], -maxVal, rLength);  // 指令 1
    
    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);
    
    // Step 4: ReduceSum
    ReduceSum<T, true>(scalarLocal, yLocal[rowOffset], reduceTmpLocal, rLength);
    
    // Step 5: Muls（除法转乘法，直接标量操作）
    T sumVal = scalarLocal.GetValue(0);
    T invSumVal = (T)1.0 / sumVal;  // CPU 端计算 1/sum
    Muls<T>(yLocal[rowOffset], yLocal[rowOffset], invSumVal, rLength);  // 指令 2
}

// 总计：4 条指令/行，节省 broadcastBuf (480B for rLength=128)
```

---

## 场景2：广播操作（多行）

### 方案对比

**问题**：需要对多行数据执行相同的标量操作（如 `x - max`、`exp / sum`）

**方案对比**：

| 方案 | API 调用 | Buffer 需求 | 推荐度 |
|-----|---------|------------|--------|
| 逐行循环 | R 次 | alignedCols×4 | ⭐⭐ |
| 单次广播（R ≤ 64） | 1 次 | alignedCols×4 | ⭐⭐⭐⭐⭐ |
| 分批广播（R > 64） | ceil(R/64) 次 | alignedCols×4 | ⭐⭐⭐⭐⭐ |

### 核心原理

**BinaryRepeatParams.src1RepStride=0 实现广播**：

```cpp
struct BinaryRepeatParams {
    uint8_t dstBlkStride;    // 单次迭代内，dst 的 block 步长
    uint8_t src0BlkStride;   // 单次迭代内，src0 的 block 步长
    uint8_t src1BlkStride;   // 单次迭代内，src1 的 block 步长
    uint8_t dstRepStride;    // 相邻迭代间，dst 的 block 步长
    uint8_t src0RepStride;   // 相邻迭代间，src0 的 block 步长
    uint8_t src1RepStride;   // =0 实现广播
};
```

**工作原理**：
- `dstRepStride = alignedCols/8`：每次迭代，dst 前进 `alignedCols` 个元素
- `src0RepStride = alignedCols/8`：每次迭代，src0 前进 `alignedCols` 个元素
- `src1RepStride = 0`：每次迭代，src1 **不前进**，重复读取相同位置

**效果**：
```
迭代 0: dst[0:cols]     = src0[0:cols]     - src1[0:cols]
迭代 1: dst[cols:2cols] = src0[cols:2cols] - src1[0:cols]  ← 重复读取
迭代 2: dst[2cols:3cols]= src0[2cols:3cols]- src1[0:cols]  ← 重复读取
```

### 分批处理

#### 方案1：逐行循环（低效）

```cpp
for (uint32_t r = 0; r < R; r++) {
    Sub(dstLocal[r * alignedCols], srcLocal[r * alignedCols], scalarLocal, alignedCols);
}
// API 调用：R 次
```

#### 方案2：单次广播（高效，R ≤ 64）

```cpp
uint64_t mask = alignedCols;
uint8_t repeatTime = R;

Sub(dstLocal, srcLocal, scalarLocal, mask, repeatTime, 
    {1, 1, 1, alignedCols/8, alignedCols/8, 0});
// API 调用：1 次
// 性能提升：R 倍
```

#### 方案3：分批广播（高效，R > 64）

```cpp
constexpr uint32_t BATCH_SIZE = 64;
uint32_t totalBatches = (R + BATCH_SIZE - 1) / BATCH_SIZE;  // ceil(R/64)

for (uint32_t batch = 0; batch < totalBatches; batch++) {
    uint32_t startRow = batch * BATCH_SIZE;
    uint8_t repeatTime = (startRow + BATCH_SIZE <= R) ? BATCH_SIZE : (R - startRow);
    uint32_t offset = startRow * alignedCols;
    
    Sub(dstLocal[offset], srcLocal[offset], scalarLocal, 
        mask, repeatTime, {1, 1, 1, alignedCols/8, alignedCols/8, 0});
}
// API 调用：ceil(R/64) 次
// 性能提升：约 64 倍
```

---

## 场景3：半精度加减法精度优化

### 问题根因

半精度（FP16=10 位尾数，BF16=7 位）两数量级差异大时会"**大数吃小数**"，Add 和 Sub 面临相同风险：

```
a = 1024.0, b = 0.0625
  Add<half>  : 1024.0     ← b 被丢弃     Sub<half>  : 1024.0     ← b 被丢弃
  Add<float> : 1024.0625  ← 正确         Sub<float> : 1023.9375  ← 正确
```

临界比值（显著退化阈值）：FP16 ≈ 2¹⁰=1024，BF16 ≈ 2⁷=128；完全丢失阈值约 2×（尾数隐含 1 位）。累加 N 次后阈值除以 √N。

### 默认策略

**spec 未明确"输入同量级"时一律升 FP32**。通用算子调用方分布未知，一旦遇到残差/累加/归一化/量化反量化即不可控。Add 和 Sub 适用同一规则，BF16 和 FP16 仅临界比值不同（见下）。

| spec 声明输入同量级？ | 推荐实现 | 理由 |
|---------------------|---------|------|
| 否（默认） | `Cast → Add/Sub<float>(in-place) → Cast` | 覆盖所有分布 |
| 是（mask 叠加、已归一化概率相加等） | 直接 `Add/Sub<half>` | 两输入本身仅 10/7 位精度，单次运算不引入额外损失；无 √N 累加放大 |

### 标准范式

`Add/Sub<float>(dst, src0, src1)` 支持 dst 与 src 别名，仅需 **K=2 份** FP32 临时空间（dst 复用 src0Fp32）：

```cpp
// Get<T>(len) 的 len 是元素数；偏移用 tensor[N]
auto src0Fp32 = tmpBuf.Get<float>(TILE);
auto src1Fp32 = src0Fp32[TILE];

// half → float 用 CAST_NONE；float → half 用 CAST_ROUND
AscendC::Cast<float, half>(src0Fp32, src0, AscendC::RoundMode::CAST_NONE, count);
AscendC::Cast<float, half>(src1Fp32, src1, AscendC::RoundMode::CAST_NONE, count);
AscendC::Add<float>(src0Fp32, src0Fp32, src1Fp32, count);   // in-place；Sub 同理
AscendC::Cast<half, float>(dst, src0Fp32, AscendC::RoundMode::CAST_ROUND, count);
```

代价：+3 条指令（共 4 条：2 Cast↑ + 1 Add/Sub + 1 Cast↓），+K×count×sizeof(float) UB。BF16 路径将 `half` 替换为 `bfloat16_t` 即可。

> **API 别名约束决定 K**：`Add/Sub<float>` 在 Vector 上支持 dst 与 src 别名，故 K=2；Reduce 类 API 禁止 dst==tmpBuffer，不可类比。

### Kernel 集成要点

> 升精度路径需要 K=2 份 FP32 临时 Buffer，Add/Sub<float> 支持 dst/src 别名故 dst 复用 src0Fp32。精度转换 RoundMode 详见 [api-precision.md](api-precision.md)。

---

## 性能对比

### 标量操作（单行）

| 项目 | 优化前 | 优化后 | 改善 |
|-----|--------|--------|------|
| **指令数/行** | 6 条 | 4 条 | **-33%** |
| **Buffer 大小** | 512B (rLength=128) | 32B | **-94%** |
| **UB 节省** | - | ~480B | 可用于更大 rowsPerLoop |

### 广播操作（多行）

| R (行数) | 逐行循环 | 单次广播 | 分批广播 | 性能提升 |
|---------|---------|---------|---------|---------|
| 32 | 32 次 | 1 次 | - | **32×** |
| 64 | 64 次 | 1 次 | - | **64×** |
| 100 | 100 次 | - | 2 次 | **50×** |
| 128 | 128 次 | - | 2 次 | **64×** |
| 200 | 200 次 | - | 4 次 | **50×** |

### 半精度加减法（FP16/BF16 Add/Sub）

升精度路径相对直接 `Add/Sub<half>`：+3 条指令（共 4 条）、+2×count×sizeof(float) UB。适用场景见[场景3 默认策略](#默认策略)。

### 实测示例（Softmax ARA 分支）

**场景**：R=128, alignedCols=64, FP32

| 操作 | 优化前 | 优化后 | 提升 |
|-----|--------|--------|------|
| Sub (x-max) | 128 次 | 2 次 | 64× |
| Div (exp/sum) | 128 次 | 2 次 | 64× |
| **总计** | **256 次** | **4 次** | **64×** |

---

## 适用 API

所有支持 `BinaryRepeatParams` 的二元运算 API：

| API | 用途 | 单行优化 | 多行优化 |
|-----|------|---------|---------|
| **Add** | 加法 | Adds | src1RepStride=0 |
| **Sub** | 减法 | Adds(-val) | src1RepStride=0 |
| **Mul** | 乘法 | Muls | src1RepStride=0 |
| **Div** | 除法 | Muls(1/val) | src1RepStride=0 |
| **Max** | 最大值 | - | src1RepStride=0 |
| **Min** | 最小值 | - | src1RepStride=0 |

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|-----|------|---------|
| 编译错误：mask 超限 | `mask > 64` (FP32) | 分批处理或回退循环 |
| 数据错误 | `src1RepStride` 未设置为 0 | 确认参数：`{..., 0}` |
| 部分行正确 | offset 计算错误 | `offset = startRow * alignedCols` |
| 越界崩溃 | repeatTime 计算错误 | 使用三目运算 |
| Buffer 不足 | 使用 Duplicate 方案 | 改用 Adds/Muls |
| dst == tmpBuffer | Reduce API 限制 | 使用不同 buffer |
| FP16/BF16 加减法精度丢失 | 直接 `Add/Sub<half>` 大数吃小数 | 升精度：`Cast→FP32 Add/Sub(in-place)→Cast` |
| 半精度加减法 Cast 后越界 | 临时 Buffer 不足 | 预留 `2 × count × sizeof(float)`，Add/Sub 复用 src0Fp32 |
| `Get<T>(len)` 取出长度异常 | 误把字节数当成元素数 | `len` 是元素数，不是字节数 |

---

## 检查清单

使用算术运算 API 时，确保：

**标量操作（单行）**：
- [ ] 使用 `Adds(-scalar)` 替代 `Duplicate + Sub`
- [ ] 使用 `Muls(1/scalar)` 替代 `Duplicate + Div`
- [ ] 标量除法转换为乘法（CPU 端计算 1/scalar）

**广播操作（多行）**：
- [ ] alignedCols ≤ 64 (FP32) / ≤ 128 (FP16)
- [ ] 使用 `src1RepStride = 0` 实现广播
- [ ] R > 64 时使用分批处理
- [ ] offset 计算正确：`offset = startRow * alignedCols`

**半精度加减法（FP16/BF16 Add/Sub）**：
- [ ] 默认升精度；仅当 spec 明确"输入同量级"时才允许直接 `Add/Sub<half>`
- [ ] 临时 Buffer 预留 `K × count × sizeof(float)`，`Add/Sub<float>` 支持别名故 K=2，dst 复用 src0Fp32（in-place）
- [ ] `Get<T>(len)` 的 len 是元素数；偏移用 `tensor[N]`
- [ ] Cast 方向：`half→float` 用 `CAST_NONE`，`float→half` 用 `CAST_ROUND`

---

## 参考资料

- [BinaryRepeatParams 结构体](../../../asc-devkit/docs/api/context/BinaryRepeatParams.md)
- [Adds API](../../../asc-devkit/docs/api/context/Adds.md)
- [Muls API](../../../asc-devkit/docs/api/context/Muls.md)
- [Sub API](../../../asc-devkit/docs/api/context/Sub.md)
- [Div API](../../../asc-devkit/docs/api/context/Div.md)
