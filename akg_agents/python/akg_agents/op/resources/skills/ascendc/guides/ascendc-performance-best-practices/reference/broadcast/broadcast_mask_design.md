# Broadcast & Mask Operations 优化设计

## 1. 优化目标

Broadcast（广播）与 Mask（掩码）是算子实现中高频出现的内存访问模式。Naive 实现通常采用逐元素标量处理或运行时条件判断，导致：

- **标量-向量混合操作**：标量输入（如学习率 lr）在循环内反复参与标量运算，无法利用 Vector 单元全带宽。
- **运行时广播模式判断**：BA/AB 广播模式在 Kernel 内通过运行时条件分支处理，引入分支预测失败开销。
- **低精度/低效的 mask 表示**：若算子原型要求 float mask（0.0/-inf），占用 4B/元素；改为 bool mask（1B/元素）可降低 75% 带宽。若原型已固定为 float，应在 Kernel 侧将 float mask Cast 为 bool 后复用，后续计算仍享内存效率收益。
- **GM 多核地址冲突**：多核并行访问同一 512B 区域时被串行处理，MTE2/MTE3 时间异常升高。

本优化通过高阶 API（SelectWithBytesMask、SoftMax）、编译期广播模式分发、标量 Duplicate 广播、GM 地址冲突规避等手段，将 broadcast/mask 场景的性能和内存效率最大化。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| mask 内存占用 | float (4B/元素) | bool (1B/元素)，原型支持或 Kernel 内 Cast 后复用 | 带宽降低 75% |
| 标量运算 | 标量-向量混合 (Muls) | 向量-向量统一 (Duplicate+Mul) | Vector 单元全带宽利用 |
| 广播模式判断 | 运行时 if-else | 编译期 `if constexpr` | 零运行时分支开销 |
| mask 语义 | 乘法 mask（精度损失） | SelectWithBytesMask（精确替换） | 语义清晰、精度无损 |
| 多核 GM 访问 | 同 512B 区域冲突串行 | 错位访问/行切分 | MTE2/MTE3 时间显著降低 |

> 适用算子族：`broadcast` 族所有含 mask 输入、广播维度、条件选择的变体，如 `scaled_masked_softmax`、`masked_scatter_with_position`、`apply_adagrad_d` 等。

## 2. 架构概览

### 2.1 存储层级与数据流

Broadcast/Mask 场景的数据流：src 和 mask 经 MTE2 从 GM 搬入 UB，在 UB 内通过高阶 API（SelectWithBytesMask、SoftMax）完成计算，结果经 MTE3 写回 GM。核心优化包括：标量 Duplicate 广播、BA/AB 编译期模式分发、MaskOffset 偏移管理、GM 地址冲突规避。

### 2.2 Broadcast 模式分类

| 模式 | 描述 | 典型场景 | 索引计算 |
|------|------|---------|---------|
| **Scalar Broadcast** | 标量输入广播为向量 | 学习率 lr、scale 等标量 | `Duplicate` 填充后向量运算 |
| **BA 模式** | mask 在后几维广播 | mask 形状 [1, S] | `maskIdx = i % xInner` |
| **AB 模式** | mask 在前几维广播 | mask 形状 [B, 1] | `maskIdx = rowIdx` |
| **Batch Broadcast** | mask 在 batch 维广播 | causal mask [1, 1, S, S] | `maskMode |= BROADCAST_BATCH` |
| **Channel Broadcast** | mask 在 channel 维广播 | channel-wise mask | `maskMode |= BROADCAST_CHANNEL` |

### 2.3 事件同步模型

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `MTE2_V` | MTE2 搬运完成 → 允许 Vector 读取 | src/mask tile 数据就绪 |
| `V_V` | Vector 完成 → 允许 Vector 继续 | SelectWithBytesMask 内部依赖 |
| `V_MTE3` | Vector 完成 → 允许 MTE3 写回 | 计算完成，可写 GM |

## 3. 关键参数配置

```cpp
// Host 侧 TilingData
struct BroadcastMaskTiling {
    uint32_t patternType;     // PATTERN_AB = 0, PATTERN_BA = 1
    uint64_t maskMode;        // bit0: batch broadcast, bit1: channel broadcast
    uint32_t padLineNum;      // 对齐后的行宽
    uint32_t alignedMaskWidth; // 对齐后的 mask 宽度
    uint32_t xInner;          // BA 模式内维大小
    uint32_t xOuter;          // BA 模式外维大小
    SoftMaxTiling softmaxTilingData;  // SoftMax 高阶 API tiling
};

// Kernel 侧 MaskOffset 结构体
struct MaskOffset {
    uint64_t batchOffset = 0;
    uint64_t channelOffset = 0;
    uint64_t lineOffset = 0;
    __aicore__ inline void NextChannel(uint64_t channelNum) {
        channelOffset = (channelOffset + 1) % channelNum;
        if (channelOffset == 0) batchOffset++;
        lineOffset = 0;
    }
    __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine) {
        return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
    }
};
```

### 3.1 对齐约束

| 数据类型 | 对齐粒度 | 说明 |
|---------|---------|------|
| FP32 | 8 元素 (32B) | width 需为 8 的倍数 |
| FP16/BF16 | 16 元素 (32B) | width 需为 16 的倍数 |
| bool (mask) | 32 元素 (32B) | mask 宽度需为 32 的倍数 |

```cpp
uint64_t alignedXBlock = AlignedBytes / xDtypeSize;   // 32 / sizeof(T)
uint64_t xPaddingNum = (width % alignedXBlock) ? (alignedXBlock - width % alignedXBlock) : 0;
uint64_t alignedMaskBlock = AlignedBytes / BOOL_SIZE; // 32 / 1 = 32
uint64_t maskPaddingNum = (width % alignedMaskBlock) ? (alignedMaskBlock - width % alignedMaskBlock) : 0;
```

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

```cpp
// 阶段 1：标量-向量混合运算（标量 lr 直接参与）
lrScalar = lrGm.GetValue(0);
AscendC::Muls(lrMulGradLocal, gradLocal, this->lrScalar, this->tileSize);

// 阶段 2：运行时广播模式判断（2D 简单处理）
uint32_t maskRow = (this->maskDim0 == 1) ? 0 : row;
uint32_t maskCol = (this->maskDim1 == 1) ? 0 : col;
uint32_t maskIdx = maskRow * this->maskDim1 + maskCol;

// 阶段 3：float mask 乘法（精度损失，语义不清）
AscendC::Muls(scaledLocal, xLocal, scale, tileLength);
AscendC::Add(scaledLocal, scaledLocal, maskLocal, tileLength);  // mask 是 float 类型

// 阶段 4：手动 softmax（无高阶 API）
AscendC::ReduceMax(maxLocal, scaledLocal, sharedLocal, tileLength);
float rowMax = maxLocal.GetValue(0);
AscendC::Duplicate(maxLocal, rowMax, tileLength);
AscendC::Sub(expLocal, scaledLocal, maxLocal, tileLength);
AscendC::Exp(expLocal, expLocal, tileLength);
AscendC::ReduceSum(sumLocal, expLocal, sharedLocal, tileLength);
float rowSum = sumLocal.GetValue(0);
float invSum = 1.0f / rowSum;
AscendC::Muls(outLocal, expLocal, invSum, tileLength);

// 阶段 5：多核同地址访问（冲突串行）
for (int i = 0; i < loopOneCore; i++) {
    DataCopy(dst, src[i * blockSize], blockSize);  // 所有核访问相同区域
}
```

### 4.2 optimized 版本（优化后）

```cpp
// === Variant A: 标量 Duplicate 广播 ===
// 标量 lr 先 Duplicate 为向量，再与 grad 做向量-向量乘法，避免标量-向量混合运算
float lrScalar = lrGm.GetValue(0);
AscendC::Duplicate(lrLocal, lrScalar, tileSize);
AscendC::Mul(dstLocal, gradLocal, lrLocal, tileSize);

// === Variant B: BA/AB 编译期模式分发 ===
if constexpr (PATTERN_TYPE == PATTERN_AB) {
    if (maskGm[rowidx] == true) { /* AB 模式 */ }
} else {
    if (maskGm[i % xInner] == true) { /* BA 模式 */ }
}

// === Variant C/D/F: SelectWithBytesMask 精确 mask 应用 ===
LocalTensor<uint8_t> maskTmpBuf = this->sharedBuffer.template Get<uint8_t>();
SelectWithBytesMaskShapeInfo shapeInfo;
shapeInfo.firstAxis = this->lineNum;
shapeInfo.srcLastAxis = this->paddedHeadDim_;
shapeInfo.maskLastAxis = this->paddedHeadDim_;
SelectWithBytesMask(tmpOutLocal, tmpOutLocal, MASK_VALUE, maskLocal, maskTmpBuf, shapeInfo);

// === Variant E: SoftMax 高阶 API ===
SoftMaxTiling softmaxTilingData = tilingData.softmaxTilingData;
SoftMaxShapeInfo softmaxShapeInfoData = {
    static_cast<uint32_t>(lines),
    static_cast<uint32_t>(tilingData.padLineNum),
    static_cast<uint32_t>(lines),
    static_cast<uint32_t>(tilingData.width),
};
SoftMax<float, false, false>(dstTensor, srcTensor, sharedBuffer, softmaxTilingData, softmaxShapeInfoData);

// === Variant D: MaskOffset 广播偏移管理 ===
struct MaskOffset {
    uint64_t batchOffset = 0;
    uint64_t channelOffset = 0;
    uint64_t lineOffset = 0;
    __aicore__ inline void NextChannel(uint64_t channelNum) {
        channelOffset = (channelOffset + 1) % channelNum;
        if (channelOffset == 0) batchOffset++;
        lineOffset = 0;
    }
    __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine) {
        return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
    }
};

// === GM 地址冲突规避 ===
// 错位访问
for (int i = 0; i < loopOneCore; i++) {
    int newProgress = (i + GetBlockIdx()) % loopOneCore;
    DataCopy(dst, src[newProgress * blockSize], blockSize);
}
```

## 5. 从 naive 到 broadcast_mask 的关键修改点

| 修改项 | naive（优化前） | broadcast_mask（优化后） |
|--------|---------------|------------------------|
| 标量运算 | 标量-向量混合 (Muls) | Duplicate 广播为向量后统一向量运算 |
| 广播模式判断 | 运行时 if-else | 编译期 `if constexpr` 零开销分支 |
| mask 数据类型 | float (4B/元素) | bool (1B/元素)，原型支持或 Kernel 内 Cast 后复用 |
| mask 应用方式 | 乘法 mask（精度损失） | SelectWithBytesMask（精确替换） |
| softmax 实现 | 手动 ReduceMax+Exp+ReduceSum | SoftMax 高阶 API（性能提升 10-20%） |
| mask 偏移计算 | 简单 2D 索引 | MaskOffset 结构体支持 batch/channel 广播 |
| 多核 GM 访问 | 同区域串行 | 错位访问或行切分避免 512B 冲突 |

## 6. 注意事项 / 约束

1. **SelectWithBytesMask 语义**：当 mask 对应位置为 true 时，dst 取 value；否则取 src。mask=true 的位置应被替换为 MASK_VAL（如 -10000.0），在 softmax 中会变成接近 0 的概率。

2. **广播模式识别在 Host 端完成**：`CanBroadcastBAOrAB` 函数在 Host 侧识别广播模式，通过 tiling data 的 `PATTERN_TYPE` 传递给 Kernel。Kernel 内使用模板参数实现编译期分支。

3. **对齐约束**：所有数据搬运和 Vector 计算必须满足 32B 对齐。FP32 需 8 元素对齐，FP16/BF16 需 16 元素对齐，bool mask 需 32 元素对齐。未对齐时使用 DataCopyPad 自动填充。

4. **SoftMax API 的临时 buffer**：使用 SoftMax 高阶 API 需要额外 UB 空间存放临时数据。可通过 `GetSoftMaxMaxTmpSize` 查询所需大小，并与 SelectWithBytesMask 的共享 buffer 精细复用。

5. **GM 地址冲突规避**：
   - 数据行宽 ≤ 512B 时冲突尤其严重
   - 错位访问需配合 `SyncAll` 全核同步
   - 行切分替代列切分可天然避免冲突，但可能导致尾行负载不均

6. **maskMode 字段定义**：bit0 表示 batch 广播，bit1 表示 channel 广播。当 batch != maskBatch 时设置 batch 广播；当 channel != maskChannel 时设置 channel 广播。

7. **精度与性能的平衡**：SoftMax 高阶 API 内部使用 FP32 中间计算保证数值稳定性，即使输入/输出是 FP16。

## 7. 常见问题与解决方案

### Q1: SelectWithBytesMask 与 float mask (Add) 有何区别？

naive 实现使用 `Add(scaledLocal, scaledLocal, maskLocal, tileLength)`，要求 mask 是 float 类型且值为 0/-inf。这种方式：
- 内存占用高（4B/元素）
- 语义不清晰（通过加法实现条件选择）
- 精度可能受乘法 mask 影响

SelectWithBytesMask 使用 bool 类型 mask（1B/元素），语义清晰（条件选择），内存占用降低 75%。若算子原型固定为 float mask，应在 Kernel 侧将 float mask Cast 为 bool 后复用，后续计算仍享内存效率收益。

### Q2: 如何处理跨 batch/channel 的 mask 广播？

使用 `MaskOffset` 结构体管理复杂的 mask 偏移计算：
```cpp
MaskOffset offset;
offset.GetOffset(realBatch, realChannel, realLine);  // 计算当前位置的 mask 偏移
offset.NextChannel(channelNum);  // 切换到下一个 channel
```

`CopyMaskIn` 函数处理多种边界情况：当当前 batch 和结束 batch 相同时，只需在一个 channel 内处理；不同时需要跨 batch 处理。

### Q3: BA/AB 模式识别失败时如何降级？

若 Host 端无法识别为 BA 或 AB 模式（如更复杂的广播模式），应降级为通用逐元素索引计算，或预先在 Host 端展开 mask 为完整形状。

### Q4: GM 地址冲突如何诊断？

Profiling 显示 `aiv_mte2_time` 或 `aiv_mte3_time` 异常高时，检查：
- 多核是否访问同一 512B 区域
- 数据行宽是否 ≤ 512B
- 是否可通过行切分或错位访问优化

## 8. 选型决策与自检清单

### 8.1 选型决策

```
if (算子包含 mask 输入或广播维度):
    → 启用 broadcast_mask 优化
    → mask 使用 bool 类型（1B/元素）；若原型为 float，Kernel 侧 Cast 为 bool 后复用
    → 标量输入通过 Duplicate 广播为向量
    → Host 端识别 BA/AB 广播模式，编译期分发
    → mask 应用使用 SelectWithBytesMask 高阶 API
    → softmax 计算使用 SoftMax 高阶 API
    → 多核场景检查 GM 地址冲突，必要时错位访问或行切分
else:
    → 标准向量运算即可
```

### 8.2 自检清单

- [ ] mask 数据类型为 bool（1B/元素）或 Kernel 内已将 float Cast 为 bool 复用
- [ ] 标量输入通过 `Duplicate` 广播为向量后参与运算
- [ ] BA/AB 广播模式在 Host 端识别，Kernel 内使用 `if constexpr` 编译期分支
- [ ] mask 应用使用 `SelectWithBytesMask`，非乘法 mask 或 Add
- [ ] SoftMax 计算使用高阶 API，非手动 ReduceMax+Exp+ReduceSum
- [ ] 所有数据搬运满足 32B 对齐，未对齐时使用 DataCopyPad
- [ ] 多核场景检查 GM 地址冲突，行宽 ≤ 512B 时启用错位访问或行切分
- [ ] MaskOffset 正确管理 batch/channel 广播偏移
- [ ] SoftMax 临时 buffer 与 SelectWithBytesMask 共享 buffer 精细复用
- [ ] 精度校验通过：与 naive 实现对比，误差 < 1e-5（FP32）或 < 1e-3（FP16）
