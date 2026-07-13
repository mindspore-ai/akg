# Tail Block / Data Alignment 优化设计

## 1. 优化目标

在 Convolution、Pooling、Gather 等算子中，输入数据量往往不能被核心数或 tile 大小整除，产生尾块（tail block）。Naive 实现采用固定 tile 处理，导致尾块访问越界、数据竞争或负载不均衡。

本优化通过双轨制计算、向量化尾块处理、多维度 Tiling 尾块分离等手段，确保在所有场景下实现近完美的负载均衡和正确的边界处理。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| 尾块处理 | 简单截断，可能越界 | 双轨制 norm/tail 核分离 | 避免数据竞争和越界访问 |
| 尾块计算 | 标量循环处理 | GatherMask 向量化处理 | 一次处理 64 个元素，提升 4-8 倍 |
| 多核负载 | 尾核空闲或超载 | 动态 costLimit + 容忍度机制 | 非整除场景性能提升 10-30% |
| 内存对齐 | 未考虑对齐 | 32B/64B 对齐 + DataCopyPad | 避免非对齐访问性能损失 |


## 2. 架构概览

### 2.1 存储层级与数据流

Tail Block 处理的数据流：输入数据经 MTE2 从 GM 搬入 UB/L1，分为正常块（normBlock）和尾块（tailBlock）。正常核处理 `defaultUbFactor` 整数倍的数据量，尾核处理剩余数据量。尾块使用 GatherMask 向量化处理或 DataCopyPad 对齐填充，结果经 MTE3 写回 GM，多核竞争场景使用 AtomicAdd 避免数据竞争。

### 2.2 双轨制计算模型

Host 侧将数据分为正常核（normCore）和尾核（tailCore）两组：
- **正常核**：处理 `defaultUbFactor` 整数倍的数据量
- **尾核**：处理剩余数据量，单独计算尾块循环次数和尾块大小

Kernel 侧根据 `blockIdx_` 判断自己是正常核还是尾核，选择对应的循环参数。

### 2.3 事件同步模型

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `MTE2_V` | MTE2 搬运完成 → 允许 Vector 读取 | 主数据 tile 就绪 |
| `V_MTE3` | Vector 完成 → 允许 MTE3 写回 | 尾块计算完成，可写 GM |
| `PIPE_V` | Vector 管道屏障 | 向量化指令间数据依赖 |

## 3. 关键参数配置

```cpp
// Host 侧 TilingData
struct TailBlockTiling {
    uint32_t normBlockLoop;           // 正常核循环次数
    uint32_t normBlockTailLoopSize;   // 正常核最后循环大小
    uint32_t tailBlockLoop;           // 尾核循环次数
    uint32_t tailBlockTailLoopSize;   // 尾核最后循环大小
    uint32_t defaultValueUsedCoreNum; // 正常核数量
    uint32_t defaultUbFactor;         // 默认 UB 处理粒度
};
```

### 3.1 Tile 大小选取原则

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `defaultUbFactor` | 64 / 128 / 256 | 向量化处理粒度，需对齐 Vector 单元宽度 |
| `alignCoef` | 32B / 64B | 数据类型相关对齐系数（FP16=32B, FP32=64B） |

**Ascend 对齐约束**：

- Global Memory 访问需 32B 对齐
- Vector 指令最优访问粒度为 32B（FP16）或 64B（FP32）
- Cube 矩阵乘需 16x16 或 32x32 对齐

```cpp
// 对齐计算
uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);  // 32B / sizeof(T)
uint32_t alignedSize = (size + numPerBlock - 1) / numPerBlock * numPerBlock;
```

### 3.2 内存预算

尾块处理需要额外的 pattern buffer（用于 GatherMask）和索引 buffer：
- `tmpPattern` buffer：约 64B（uint32_t/uint16_t pattern）
- `indexBuf` buffer：6 个索引 × tile 大小（用于 adaptive pooling）

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

```cpp
// 固定 tile 处理，无尾块特殊处理
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
// 问题：最后一 tile 可能越界；尾核负载不均
```

### 4.2 optimized 版本（优化后）：双轨制尾块处理

```cpp
// Host 侧：计算 norm/tail 核参数
normBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleDefaultValues_, defaultUbFactor_);
normBlockTailLoopSize_ = normCoreHandleDefaultValues_ - defaultUbFactor_ * (normBlockLoop_ - 1);
tailBlockLoop_ = Ops::Base::CeilDiv(tailCoreHandleDefaultValues, defaultUbFactor_);
tailBlockTailLoopSize_ = tailCoreHandleDefaultValues - defaultUbFactor_ * (tailBlockLoop_ - 1);

// Kernel 侧：根据 blockIdx 选择参数
loop_ = tilingData_.normBlockLoop;
tailLoopSize_ = tilingData_.normBlockTailLoopSize;
if (blockIdx_ == tilingData_.defaultValueUsedCoreNum - 1) {
    loop_ = tilingData_.tailBlockLoop;
    tailLoopSize_ = tilingData_.tailBlockTailLoopSize;
}

// 主循环
for (uint32_t i = 0; i < loop_; i++) {
    bool isLastLoop = (i == loop_ - 1);
    uint32_t currentTileSize = isLastLoop ? tailLoopSize_ : defaultUbFactor_;
    
    // 使用 DataCopyPad 处理非对齐尾块
    if (isLastLoop && currentTileSize != defaultUbFactor_) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentTileSize * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[...], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[...], currentTileSize);
    }
    
    Compute(dataLocal, currentTileSize);
    CopyOut(dataLocal, currentTileSize);
}
```

### 4.3 GatherMask 向量化尾块处理

```cpp
// 对于最后一个输出点的尾块，使用 GatherMask 进行数据重排
if constexpr (std::is_same_v<T, float>) {
    LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
    int32_t preLeftShift = numPerBlock + lastLeftShift;
    bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
    GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], 
               bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
}
// 避免 atomic 操作开销，正确处理尾块数据布局
```

### 4.4 非对齐数据的原子操作处理

```cpp
// 针对数据非 32B 对齐的情况，使用 AtomicAdd 避免数据竞争
uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
uint64_t mask[2] = {mask0, 0};
Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
SetAtomicAdd<T>();
DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
SetAtomicNone();
```

## 5. 从 naive 到 tail_block 的关键修改点

| 修改项 | naive（优化前） | tail_block（优化后） |
|--------|---------------|---------------------|
| 核类型 | 所有核处理相同数据量 | 双轨制：normCore + tailCore |
| 尾块处理 | 简单截断或越界访问 | GatherMask 向量化 / DataCopyPad 对齐 |
| 数据竞争 | 无保护，可能写冲突 | AtomicAdd / 边界跳过 |
| 负载均衡 | 固定分配 | 动态 costLimit + 容忍度机制 |
| 内存对齐 | 未考虑 | 32B/64B 对齐 + padding |
| 索引计算 | 运行时除法/取模 | IndexBuffer 预计算复用 |

## 6. 注意事项 / 约束

1. **GatherMask 的 pattern buffer 类型**。float 类型用 uint32_t pattern，half/bfloat16_t 用 uint16_t pattern。

2. **AtomicAdd 的性能开销**。虽然保证正确性，但 atomic 操作有一定性能损失，优先使用 GatherMask 避免竞争。

3. **尾块大小必须 > 0**。若尾块大小为 0，该核应直接返回，避免空循环。

4. **多维度 Tiling 的尾块**。每个维度（Channel、Spatial、Batch）都需独立计算尾块参数。

5. **重叠检测**。Adaptive Pooling 中 kernel 可能重叠，需检测并设置 `isOverLap` 标志，使用 atomic add 保证正确性。

6. **对齐填充的数据正确性**。SkipPad 策略需确保 padding 区域的 0 不会被错误计入后续计算（如方差）。

7. **32B 对齐是硬性要求**。未对齐的 GM 访问可能导致硬件异常或性能下降。

## 7. 实施常见问题与解决方案

### Q1: 尾核处理量远小于正常核，导致尾核很快完成但需等待正常核
**A**: 使用容忍度机制（tolerance ratio），允许轻微超载以减少碎片。`costLimit` 动态计算为"剩余总代价 / 剩余核数"。

### Q2: GatherMask 的 pattern 计算错误导致数据重排异常
**A**: 确保 `preLeftShift` 和 `lastLeftShift` 计算正确，pattern 值为 `(1u << preLeftShift) - (1u << lastLeftShift)`。

### Q3: DataCopyPad 的 padding 值污染后续计算
**A**: 使用 `SkipPadSubMean` 等策略跳过 padding 区域，或在计算前用 `Duplicate` 填充 padding 区域为中性值（如 0 或 MASK_VALUE）。

### Q4: 多核场景下尾块原子加导致结果非确定性
**A**: 使用确定性模式（deterministic mode），反向处理并检测边界索引冲突，跳过冲突索引由前一个核处理。

## 8. 场景用例

### 8.1 各算子族实例

**Elementwise** `[1000, 128]`, tileRows=256，尾块 232 行，策略 1：

```cpp
uint32_t loopCount = (rows + tileRows - 1) / tileRows;
for (uint32_t loop = 0; loop < loopCount; loop++) {
    uint32_t curRows = (loop == loopCount - 1)
                           ? (rows - loop * tileRows) : tileRows;
    uint32_t curLen = curRows * cols;
    DataCopy(xLocal, xGm[loop * tileRows * cols], curLen);
    Adds(yLocal, xLocal, 1.0f, curLen);
    DataCopy(yGm[loop * tileRows * cols], yLocal, curLen);
}
```

**MatMul** `M=1023`, blockM=128，尾块 127 行，策略 2+1：

```cpp
// Host：双轨制分核
uint32_t mPerCore = CeilDiv(M, coreNum);
// Kernel：根据 blockIdx 确定 M 范围
uint32_t localM = (blockIdx < coreNum - 1) ? mPerCore : M - blockIdx * mPerCore;
uint32_t loopCount = (localM + blockM - 1) / blockM;
for (uint32_t loop = 0; loop < loopCount; loop++) {
    uint32_t curM = (loop == loopCount - 1) ? (localM - loop * blockM) : blockM;
}
```

**FlashAttention** `seqLen=1025`, tileSeq=1024，尾块 1 行，策略 1+3：

```cpp
uint32_t kvLoopCount = (kvSeqLen + kvTileSize - 1) / kvTileSize;
for (uint32_t kvLoop = 0; kvLoop < kvLoopCount; kvLoop++) {
    uint32_t curKvLen = (kvLoop == kvLoopCount - 1)
                            ? (kvSeqLen - kvLoop * kvTileSize) : kvTileSize;
    if (curKvLen % 16 != 0) {  // Cube 输入需对齐
        DataCopyPad(kLocal, kGm[...], copyParams, padParams);
    }
    if (isCausal && qLoop == kvLoopCount - 1) {
        ApplyCausalMaskPartial(..., curKvLen);  // 尾块 causal mask
    }
}
```

---

## 9. 选型决策与自检清单

### 9.1 选型决策

```
if (算子包含迭代循环 && 数据量不能被 tile 大小整除):
    → 启用 tail_block 优化
    → 计算 normBlockLoop / tailBlockLoop
    → 尾块使用 DataCopyPad 或 GatherMask
    → 多核场景使用 AtomicAdd 或确定性模式
else:
    → 标准 tile 处理即可
```

### 9.2 自检清单

- [ ] Host 侧正确计算 normBlockLoop / tailBlockLoop / normBlockTailLoopSize / tailBlockTailLoopSize
- [ ] Kernel 侧根据 blockIdx_ 正确选择 norm/tail 参数
- [ ] 尾块大小为 0 时核直接返回
- [ ] DataCopyPad 参数正确（blockCount, blockLen, srcStride, dstStride）
- [ ] GatherMask pattern 类型与数据类型匹配（float→uint32_t, half→uint16_t）
- [ ] 32B/64B 对齐约束满足
- [ ] AtomicAdd 在 SetAtomicAdd / SetAtomicNone 配对内使用
- [ ] 确定性模式下反向处理和边界检测正确
- [ ] 精度校验通过：与 naive 实现对比，误差 < 1e-5（FP32）或 < 1e-3（FP16）
