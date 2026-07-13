# DataCopy Optimization 优化设计

## 1. 优化目标

数据搬运（DataCopy）是 Ascend C 算子中最基础也最频繁的操作。Naive 实现通常采用简单的逐块搬运，未充分利用硬件 DMA 引擎的带宽潜力，导致：

- **非对齐访问惩罚**：数据未按设备最优粒度对齐时，DMA 效率大幅下降。以 Ascend 910B 实测为例，32B 对齐的带宽约为 512B 对齐的 70%，不同设备的最优对齐值需实测确认。
- **小批量多次搬运**：循环内每次迭代独立 DataCopy，DMA 启动开销累积。
- **格式转换额外开销**：ND→NZ 格式转换需要独立的 TransData 算子，增加一次 L1 读写往返。
- **非连续内存访问低效**：Scatter/Gather 场景下逐元素访问无法利用 SIMD 并行性。
- **内存访问模式未优化**：跨步访问（strided access）导致大量小传输（< 256B），DMA 设置成本占主导。

本优化通过 DataCopyPad 自动对齐、ND2NZ 融合格式转换、Scatter/Gather 向量化、批量 DataCopy 合并、按设备最优粒度的 GM 对齐等手段，将数据搬运效率最大化。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| 对齐访问带宽 | 未按最优粒度对齐: ~70% | 按设备最优粒度对齐 | 带宽提升最高 30%（Ascend 910B 实测） |
| DMA 启动次数 | loopCount 次 | 1 次（批量合并） | 显著降低指令发射开销 |
| ND→NZ 转换 | 独立 TransData 算子 | 搬运时融合完成 | 省去一次 L1 读写往返 |
| Scatter/Gather | 逐元素 GetValue/SetValue | 向量化 Gather/Scatter 指令 | 性能提升 5-20 倍 |
| 非连续列提取 | 搬运全部数据后裁剪 | blockLen+srcStride 一次提取 | 减少无效数据搬运 |

> 适用算子族：`conversion` 族所有涉及数据搬运、格式转换、内存布局重排的变体，如 `transpose`、`concat`、`split`、`gather`、`scatter`、`nd2nz` 等。

## 2. 架构概览

### 2.1 存储层级与数据流

DataCopy 优化覆盖数据从 GM 到 UB/L1 的全链路：通过 DataCopyPad 实现非对齐数据自动填充对齐；通过 Nd2NzParams 在搬运时融合 ND→NZ 格式转换；通过 DataCopyExtParams 的 blockLen+srcStride 实现非连续列一次提取；通过批量合并减少 DMA 启动次数；通过 Gather/Scatter 偏移表实现向量化离散访问；通过按设备最优粒度的 GM 对齐（如 Ascend 910B 为 512B）最大化带宽利用率。

### 2.2 优化策略矩阵

| 场景 | 优化策略 | 核心 API / 参数 |
|------|---------|----------------|
| 非对齐数据搬运 | DataCopyPad 自动填充 | `DataCopyPadExtParams{isPad, leftPadding, rightPadding, paddingValue}` |
| ND→NZ 格式转换 | 搬运时融合 | `Nd2NzParams{nValue, dValue, srcDValue, dstNzC0Stride, dstNzNStride}` |
| 非连续列提取 | blockLen + srcStride | `DataCopyExtParams{blockCount, blockLen, srcStride, dstStride}` |
| 批量输出合并 | 循环内累积，循环外统一写回 | 累积到 UB，循环结束后一次性 DataCopy |
| 向量化 Gather | 预计算偏移表 + Gather 指令 | `Gather(dst, src, offsetTable, 0, count)` |
| 向量化 Scatter | 索引计算 + DataCopyPad | `inputStride0_ / inputStride1_` 非连续寻址 |
| GM 带宽最大化 | 按设备最优粒度对齐（如 512B） | `AlignUp(offset, 512)` |

### 2.3 DMA 效率阈值

以下阈值基于 Ascend 910B 实测经验，不同设备的 DMA 控制器、总线宽度和缓存行大小不同，具体数值需参考对应硬件手册或实测确认。

| 传输大小 | 效率 | 说明 |
|---------|------|------|
| < 32 bytes | 极低 | 对齐开销占主导 |
| 32-256 bytes | 差 | DMA 设置成本显著 |
| 256-4096 bytes | 中等 | 大多数场景可接受 |
| > 4096 bytes | 好 | 总线带宽充分利用 |
| > 65536 bytes | 极佳 | 接近峰值吞吐 |

## 3. 关键参数配置

```cpp
// DataCopyExtParams 结构（多维批量传输）
struct DataCopyExtParams {
    uint16_t blockCount;   // 块数量
    uint32_t blockLen;     // 每块字节数
    uint32_t srcStride;    // 源地址块间步长（字节）
    uint32_t dstStride;    // 目的地址块间步长（字节）
    uint32_t reserved;     // 保留
};

// DataCopyPadExtParams 结构（自动填充）
template <typename T>
struct DataCopyPadExtParams {
    bool isPad;            // 是否启用填充
    uint8_t leftPadding;   // 左侧填充字节数
    uint8_t rightPadding;  // 右侧填充字节数（最大 255）
    T paddingValue;        // 填充值
};

// Nd2NzParams 结构（ND→NZ 格式转换）
struct Nd2NzParams {
    uint32_t nValue;       // N 维大小
    uint32_t dValue;       // D 维大小
    uint32_t srcDValue;    // 源 D 维步长
    uint32_t dstNzC0Stride; // 目的 NZ C0 维步长（需对齐：fp16 为 16，fp8 为 32）
    uint32_t dstNzNStride; // 目的 NZ N 维步长
};
```

### 3.1 对齐参数计算

| 数据类型 | 32B 对齐元素数 | 512B 对齐元素数 | 说明 |
|---------|--------------|----------------|------|
| FP32 | 8 | 128 | 地址/长度需为 8 的倍数 |
| FP16/BF16 | 16 | 256 | 地址/长度需为 16 的倍数 |
| INT8 | 32 | 512 | 地址/长度需为 32 的倍数 |

```cpp
// 32B 对齐
uint32_t align32 = (size + 31) / 32 * 32;  // 或 CeilAlign(size, 32)
// 按设备最优粒度对齐（如 Ascend 910B 为 512B）
uint32_t align512 = (offset + 511) / 512 * 512;
```

### 3.2 Padding 限制

- `rightPadding` 为 `uint8_t`，最大补零 255 字节
- `blockLen` 单位为字节，需注意数据类型大小转换
- 补零后的数据参与后续计算，需确保补零不影响算法正确性（如 ReduceSum 场景补零安全，但 ReduceMax 可能受影响）

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

```cpp
// 阶段 1：简单 DataCopy，无对齐处理
AscendC::DataCopy(xLocal, xGm[offset], this->tileSize);

// 阶段 2：ND→NZ 需要独立 TransData 算子
AscendC::DataCopy(l1Tensor, gmTensor, size);
TransData(l1Tensor, nzTensor, ...);  // 额外算子

// 阶段 3：循环内每次独立 DataCopy 写回
for (int i = 0; i < loopCount; i++) {
    Compute();
    DataCopy(gm[offset], ub, size);  // 每次 16 个元素
}

// 阶段 4：逐元素 Scatter 写入
for (int64_t i = 0; i < loadCount; i++) {
    IndexT idx0 = indLocal.GetValue(i * INDICES_LAST_DIM);
    IndexT idx1 = indLocal.GetValue(i * INDICES_LAST_DIM + 1);
    int64_t gmOffset = idx0 * inputStride0_ + idx1 * inputStride1_;
    // 逐行搬出，每次只搬 1 行
    DataCopy(inputGm_[gmOffset], updLocal[i * updateRowElements_], updateDimSize_);
}

// 阶段 5：逐元素 Gather 读取
for (uint32_t j = 0; j < idxGatherDim; j++) {
    int32_t gatherIdx = idxLocal.GetValue(idxRowOffset + j);
    float value = xLocal.GetValue(xRowOffset + gatherIdx);
    yLocal.SetValue(yRowOffset + j, value);
}

// 阶段 6：非连续列提取——搬运全部数据后裁剪
DataCopy(fullLocal, gmSrc, fullSize);  // 搬运全部数据
for (int i = 0; i < rows; i++) {
    ExtractColumn(local[i], fullLocal[i], colStart, colLen);  // UB 上裁剪
}
```

### 4.2 optimized 版本（优化后）

```cpp
// === Variant A: DataCopyPad 自动对齐填充 ===
uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
dataCopyParams.blockLen = info.s2dealNum;
dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
dataCopyParams.dstStride = 0;
DataCopyPadExtParams<bool> padParams{true, 0,
    static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 0};
DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);

// === Variant B: ND→NZ 融合格式转换 ===
template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor,
    const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue) {
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue;
    gm2L1Nd2NzParams.srcDValue = srcDValue;
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;  // fp16 对齐 16
    gm2L1Nd2NzParams.dstNzNStride = 1;
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

// === Variant C: 批量 DataCopy 合并输出 ===
LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();
for (uint32_t i = 0; i < loopCount; i++) {
    DealBmm1ResBaseBlock(info, nInt32Out, ...);  // 不包含 DataCopy
}
outputQue2.EnQue(nInt32Out);
outputQue2.DeQue<int32_t>();
uint32_t dealRowCount = (loopCount - 1) * gSplitSize + tailSplitSize;
DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);  // 一次性写回

// === Variant D: 预计算偏移表 + Gather 提取 ===
// Init 阶段预计算（只执行一次）
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    for (uint32_t j = 0; j < N_; j++)
        postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
    curOffset += nSquare;
}
// 后续通过 Gather 提取
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
Gather(hPostBuff_, matmulRes_, postOffsetBuf_, 0, lenT * N_);

// === Variant E: Stride 非连续列提取 ===
DataCopyExtParams copyParams{
    static_cast<uint16_t>(ubFactor),
    static_cast<uint32_t>(RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
    static_cast<uint32_t>(ROPE_LENGTH * sizeof(KV_DTYPE)),
    0, 0};
DataCopyPad(xLocal, kvGm[kvGlobalMemoryOffset], copyParams, padParams);

// === Variant F: GM 按设备最优粒度对齐带宽优化 ===
// 以下数据为 Ascend 910B 实测结果，不同设备的最优对齐粒度可能不同。
uint32_t offset = AlignUp(rawOffset, 512);  // 910B 上 512B 为最优粒度
DataCopy(ubTensor, gmTensor[offset], dataSize);
// 实测带宽对比（GM→UB，Ascend 910B）：
// 按最优粒度（512B）对齐: ~100% 带宽效率
// 256B 对齐: ~90% 带宽效率
// 32B 对齐:  ~70% 带宽效率（最差情况）
```

## 5. 从 naive 到 datacopy_optimization 的关键修改点

| 修改项 | naive（优化前） | datacopy_optimization（优化后） |
|--------|---------------|-------------------------------|
| 非对齐搬运 | 简单 DataCopy（可能非对齐） | DataCopyPad 自动填充对齐 |
| ND→NZ 转换 | 独立 TransData 算子 | 搬运时融合 Nd2NzParams |
| 循环输出 | 每次迭代独立 DataCopy | 循环内累积，循环外统一写回 |
| Scatter 写入 | 逐元素 SetValue | 向量化 DataCopyPad 逐行写出 |
| Gather 读取 | 逐元素 GetValue | 预计算偏移表 + Gather 指令 |
| 非连续列提取 | 搬运全部后裁剪 | blockLen+srcStride 一次提取 |
| GM 地址对齐 | 无特殊处理 | 按设备最优粒度对齐最大化带宽 |
| DMA 传输大小 | 多次小传输（< 256B） | 合并为大传输（> 4096B） |

## 6. 注意事项 / 约束

1. **DataCopyPad rightPadding 限制**：`rightPadding` 为 `uint8_t`，最大补零 255 字节。超过此限制需手动分块处理。

2. **补零数据的安全性**：补零后的数据参与后续计算，需确保补零不影响算法正确性。例如 ReduceSum 场景补零安全（0 不影响求和），但 ReduceMax 可能受影响（0 可能改变最大值）。

3. **ND2NZ 对齐要求**：`dstNzC0Stride` 需按数据类型对齐——fp16 为 16 元素，fp8 为 32 元素。不同数据类型对齐基数不同，需条件编译。

4. **DataCopyParams stride 为 uint16_t**：最大 65535；超限需切换到 `DataCopyExtParams`。

5. **Scatter 逐行写出 MTE3 效率低**：每次只搬 1 行，适合 update 行数较少的场景。行数多时考虑批量处理或重组数据布局。

6. **Gather 偏移表 UB 占用**：偏移表占用 UB 空间（元素数 × 4 字节），子张量数量多时占用显著。offset 必须是字节偏移且 Gather 要求源数据在 UB 中连续。

7. **GM 按设备最优粒度对齐**：Kernel 入参（包括 Workspace/Tiling）地址通常已保证对齐，开发者需关注偏移量是否保持该设备的最优对齐粒度（如 Ascend 910B 为 512B，其他设备需查阅手册或实测确认）。

8. **DMA 效率阈值**：传输大小 < 256B 时 DMA 设置成本显著；> 4096B 时总线带宽充分利用。避免过度 tiling 导致每次 DMA 传输过小。具体阈值因设备 DMA 控制器特性而异，需参考对应硬件手册。

9. **列提取时 srcStride 和 blockLen 必须满足 32B 对齐**：每次提取一个子字段需多次搬运调用。

10. **批量 DataCopy 的 UB 空间管理**：循环内累积需要额外 UB 空间，需确保总占用小于 UB 容量。首次迭代可能有额外判断逻辑。

## 7. 常见问题与解决方案

### Q1: DataCopyPad 的 padding 值如何设置？

```cpp
// 一般场景：补零
DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};

// 需要特定填充值的场景（如 mask 填充）
DataCopyPadExtParams<T> padParams{true, 0, rightPadding, MASK_VALUE};

// 不启用填充（数据已对齐）
DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
```

### Q2: ND2NZ 转换时不同数据类型的对齐基数？

| 数据类型 | C0 对齐基数 | 计算公式 |
|---------|-----------|---------|
| FP16 | 16 元素 | `(nValue + 15) >> 4 << 4` |
| FP8 | 32 元素 | `(nValue + 31) >> 5 << 5` |

### Q3: 批量 DataCopy 合并时如何处理尾块？

```cpp
uint32_t dealRowCount = (loopCount - 1) * gSplitSize + tailSplitSize;
DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);
```

尾块大小 `tailSplitSize` 通常小于标准块 `gSplitSize`，需在 Host 侧 tiling 时计算并传递。

### Q4: Gather 偏移表如何预计算？

偏移表在 `Init` 阶段只计算一次，后续迭代复用：
```cpp
// Init 阶段
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    // ...
}
// Process 阶段（每次迭代）
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
```

### Q5: 如何诊断 DMA 效率问题？

Profiling 关注以下指标：
- `aiv_mte2_time` / `aiv_mte3_time` 异常高 → 检查对齐和传输大小
- 带宽利用率低 → 检查是否大量小传输（< 256B）
- 大量 DMA 操作 → 检查是否可合并为批量传输

## 8. 选型决策与自检清单

### 8.1 选型决策

```
if (算子涉及数据搬运或格式转换):
    → 启用 datacopy_optimization
    
    if (数据未对齐 32B):
        → 使用 DataCopyPad 自动填充
    
    if (需要 ND→NZ 格式转换):
        → 使用 Nd2NzParams 搬运时融合转换
    
    if (循环内多次 DataCopy 写回):
        → 合并为批量 DataCopy，循环外统一写回
    
    if (需要按索引提取子张量):
        → 预计算偏移表 + Gather 指令
    
    if (需要非连续列提取):
        → 使用 blockLen + srcStride 一次提取
    
    if (多核并行访问 GM):
        → 确保按设备最优粒度对齐，必要时错位访问或行切分
    
    if (传输大小 < 256B):
        → 考虑合并传输或调整 tiling 策略
else:
    → 标准 DataCopy 即可
```

### 8.2 自检清单

- [ ] 所有数据搬运满足 32B 对齐，未对齐时使用 DataCopyPad
- [ ] GM 地址偏移保持设备最优对齐粒度（如 512B）以最大化带宽，不同设备需实测确认
- [ ] ND→NZ 转换使用 `Nd2NzParams`，`dstNzC0Stride` 按数据类型对齐
- [ ] 循环内多次 DataCopy 已合并为批量输出
- [ ] Gather 偏移表在 Init 阶段预计算，Process 阶段复用
- [ ] Scatter 逐行写出时评估 MTE3 效率，行数多时考虑优化
- [ ] DataCopyPad 的 `rightPadding` ≤ 255 字节
- [ ] 补零数据不影响算法正确性（ReduceMax 场景特别注意）
- [ ] `DataCopyParams` 的 stride ≤ 65535，超限使用 `DataCopyExtParams`
- [ ] 传输大小 > 256B（以 Ascend 910B 为例），避免大量小 DMA 传输，不同设备阈值可能不同
- [ ] UB 空间预算充足，批量累积不超出容量
- [ ] 精度校验通过：与 naive 实现对比，数据一致性 100%
