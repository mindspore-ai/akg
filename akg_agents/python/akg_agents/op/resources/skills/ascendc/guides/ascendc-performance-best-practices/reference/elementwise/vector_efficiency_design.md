# Vector 计算效率优化设计

## 1. 优化目标

在纯 Vector 或 Vector 主导的算子中，性能瓶颈常来自三类 UB 内部效率损失：Scalar 侧为每个 Vector API 手动计算 repeat/mask，归约场景反复使用高延迟 reduce 指令，以及连续 Vector 计算链把中间结果写回 GM 后再读回 UB。本优化通过 Counter 模式减少控制指令、低延迟指令组合替代单一高延迟归约路径、UB 内数据直通消费，**减少 Scalar 开销、Vector 指令延迟和 GM 往返搬运**。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| Scalar 控制指令 | 每条 Vector API 前手动计算 repeat/mask | Counter 模式一次设置，全程自动 | Scalar 开销降低 30-50% |
| GM 搬运次数（n 步 Vector 链） | 2n 次（每步搬入+搬出） | 2 次（首次搬入 + 最终搬出） | 搬运次数减少 n 倍 |
| 归约指令延迟 | WholeReduceSum 单条高延迟 | BlockReduceSum + WholeReduceSum 组合 | 延迟降低 20-40% |
| 指令级并行 | 循环体串行执行 | 循环展开暴露并行性 | ILP 提升 |


## 2. 架构概览

### 2.1 三类优化在 Vector PIPE 中的位置

三类优化均作用于单核 UB 内的 Vector 执行路径：

- **Counter 模式（入口）**：`SetMaskCount(dataSize)` 后一次 Vector API 调用覆盖全部数据，避免手动计算 `repeatTimes` 和 `tailSize`。
- **UB 融合链（中间）**：通过 `VECCALC` 队列将多步 Vector 计算的中间结果保留在 UB 内，不落地 GM，减少 `2n → 2` 次搬运。
- **低延迟归约（出口）**：用 `BlockReduceSum + WholeReduceSum` 组合替代单一高延迟归约路径。

### 2.2 数据流对比

| 模式 | 数据流 | GM 搬运次数 |
|------|--------|------------|
| naive（反例） | 每步 Vector 后将中间结果搬出 GM，下一步再搬入 | `2n` 次 |
| optimized（正例） | 首次搬入 UB(VECCALC)，中间结果链式消费，最终一次搬出 | `2` 次 |

## 3. 关键参数配置

```cpp
// Counter 模式参数
struct CounterModeConfig {
    uint32_t totalElements;   // 总元素数，直接传给 SetMaskCount
    uint32_t oneRepeatSize;   // 单次 repeat 处理元素数（如 128/256）
};

// UB 融合链参数
struct FusedChainConfig {
    uint32_t chainLength;     // Vector 计算链长度（如 3：Exp→Abs→Mul）
    uint32_t veccalcBufferSize; // VECCALC buffer 大小（元素数）
};

// 低延迟归约参数
struct LowLatencyReduceConfig {
    uint32_t blockSize;       // BlockReduceSum 的块大小（如 64/128）
    uint32_t numBlocks;       // 总块数 = totalElements / blockSize
};
```

### 3.1 参数选取原则

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `oneRepeatSize` | 128 (FP16) / 64 (FP32) | 与 Vector Unit 宽度对齐 |
| `chainLength` | 2-5 | 受限于 UB 容量，过长会挤占双缓冲空间 |
| `blockSize` | 64 / 128 | BlockReduceSum 的有效块大小 |

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

**Counter 反例 — 手动 repeat/mask：**
```cpp
// 每条 Vector API 前手动计算 repeat 和 tail
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;

// 主块
Add(dst, src1, src2, FULL_MASK, repeatTimes, {1, 1, 1, 8, 8, 8});

// 尾块（需要额外一次 API 调用 + mask 设置）
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst + offset, src1 + offset, src2 + offset, tailMask, 1, {1, 1, 1, 8, 8, 8});
}
```

**UB 融合反例 — 每步搬出 GM：**
```cpp
void Process() {
    // Step 1: Exp
    CopyIn();           // GM → UB
    Compute_Exp();      // UB → UB
    CopyOut();          // UB → GM

    // Step 2: Abs
    CopyIn1();          // GM → UB
    Compute_Abs();      // UB → UB
    CopyOut1();         // UB → GM
}
```

**归约反例 — 单一 WholeReduceSum：**
```cpp
// 高延迟单条指令
float sum = WholeReduceSum(src, dataSize);  // 延迟高，大数据量时明显
```

### 4.2 optimized 版本（优化后）

**Counter 模式 — 一次调用覆盖全部：**
```cpp
// Counter 模式：SetMaskCount 后一次 API 调用
SetMaskCount(dataSize);
Add(dst, src1, src2, dataSize);  // 自动处理主块+尾块，无需手动 mask
SetMaskMode(NORMAL);  // 完成后恢复 mask 模式
```

**UB 融合链 — 中间结果不落地 GM：**
```cpp
// 定义 VECCALC 队列用于中间结果
TQue<QuePosition::VECCALC, 1> midQueue;
pipe.InitBuffer(midQueue, 1, dataSize * sizeof(T));

// 仅首次搬入、最终搬出
CopyIn(inLocal, inGm, dataSize);  // GM → UB(VECCALC)

// 链式计算，中间结果保留 UB
LocalTensor<T> mid1 = midQueue.AllocTensor<T>();
Exp(mid1, inLocal, dataSize);           // Exp

LocalTensor<T> mid2 = midQueue.AllocTensor<T>();
Abs(mid2, mid1, dataSize);              // Abs

LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
Mul(outLocal, mid2, scale, dataSize);   // Mul

CopyOut(outGm, outLocal, dataSize);      // UB → GM
```

**低延迟归约 — BlockReduceSum + WholeReduceSum 组合：**
```cpp
// 两阶段归约：先 BlockReduceSum 分块，再 WholeReduceSum 汇总
// 阶段 1：BlockReduceSum，每 blockSize 个元素产生一个部分和
for (uint32_t i = 0; i < numBlocks; i++) {
    blockSums[i] = BlockReduceSum(src + i * blockSize, blockSize);
}
PipeBarrier<PIPE_V>();

// 阶段 2：WholeReduceSum 汇总部分和
float totalSum = WholeReduceSum(blockSums, numBlocks);
```

## 5. 从 naive 到 vector_efficiency 的关键修改点

| 修改项 | naive（优化前） | vector_efficiency（优化后） |
|--------|---------------|---------------------------|
| Vector API 调用方式 | 手动计算 repeatTimes + tail mask | Counter 模式：SetMaskCount + 一次调用 |
| 中间结果生命周期 | 每步搬出 GM，下步再搬入 | 保留 UB/VECCALC，链式直通消费 |
| 归约指令 | WholeReduceSum 单条高延迟 | BlockReduceSum + WholeReduceSum 组合 |
| GM 搬运次数（n 步链） | 2n 次 | 2 次 |
| Scalar 开销 | 高（repeat/mask 计算） | 低（Counter 模式自动处理） |

## 6. 注意事项 / 约束

1. **Counter 模式依赖硬件支持**：仅部分 Vector API 支持 Counter 模式，使用前需确认 API 文档。

2. **Counter 模式必须恢复 mask**：使用完 Counter 模式后，必须调用 `SetMaskMode(NORMAL)` 恢复，避免影响后续 Vector API。

3. **UB 融合链长度受 UB 容量限制**：chainLength 过长或 dtype 较宽时，中间 buffer 可能挤占双缓冲空间。需精确计算 UB 预算。

4. **BlockReduceSum 需要临时 buffer**：低延迟归约组合需要额外的 blockSums buffer，需确保 UB 有足够空间。

5. **循环展开与 Hardware Loop 的交互**：展开后的循环仍需满足 Hardware Loop 条件（`uint16_t` 迭代变量、起始 0、步长 1），避免引入 if/else。

6. **尾块处理**：Counter 模式自动处理尾块，但 Normal 模式下仍需保留尾块处理逻辑作为 fallback。

## 7. 实施常见问题与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| Counter 模式后后续 API 结果异常 | mask 未恢复为 NORMAL | 每次 Counter 模式后调用 `SetMaskMode(NORMAL)` |
| UB 融合链 UB 溢出 | chainLength 过长或 dtype 宽 | 减少 chainLength 或 tile_size；精确计算 UB 预算 |
| BlockReduceSum 结果不正确 | blockSize 未对齐 | blockSize 需对齐 Vector Unit 宽度（如 64/128） |
| 循环展开后性能下降 | 展开因子过大导致 ICache miss | 展开因子 ≤ 4，默认推荐 2x |

## 8. 选型决策与自检清单

### 8.1 选型决策

```
if (profiling 显示 Scalar 时间占比高):
    → 启用 Counter 模式
elif (profiling 显示 MTE2/MTE3 中间搬运占比高，且有多步 Vector 链):
    → 启用 UB 融合链（VECCALC 直通）
elif (profiling 显示归约延迟高):
    → 启用低延迟归约组合（BlockReduceSum + WholeReduceSum）
else:
    → 标准 Vector 实现即可
```

### 8.2 自检清单

- [ ] Counter 模式使用后在适当位置恢复 `SetMaskMode(NORMAL)`
- [ ] UB 融合链的 VECCALC buffer 大小已精确计算，不溢出
- [ ] BlockReduceSum 的 blockSize 对齐 Vector Unit 宽度
- [ ] 循环展开后尾块处理正确（`repTimes % unrollFactor`）
- [ ] 寄存器预算 ≤ 32 RegTensor
- [ ] 验证通过：与 naive 实现对比，结果一致
