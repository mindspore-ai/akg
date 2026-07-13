# Softmax 状态 Buffer 跨循环常驻优化设计

## 1. 优化目标

在 Flash Attention / Sparse Flash Attention 等场景中，online softmax 需要在 S2 方向多次循环中累积 `softmaxMax`、`softmaxSum`、`softmaxExp` 三个状态。naive 实现每次 S2 循环都重新分配/释放这些 UB buffer，引入不必要的 `PipeBarrier` 和 `InitBuffer` 开销。

本优化将三个状态 buffer **一次性分配后常驻 UB**，通过 `loop % preLoadNum` 索引实现双缓冲复用，避免每轮 S2 循环的重复分配开销，同时支持读写并行流水。

**Source operators**: `ai_infra_sparse_flash_attention_gqa`, `ai_infra_fused_infer_attention_sink`

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| S2 循环内 InitBuffer 调用 | 每轮 3 次 | 0 次 | 消除重复分配开销 |
| PipeBarrier 数量 | 每轮若干 | 显著减少 | 减少事件同步等待 |
| UB 空间利用率 | 低（频繁分配释放产生碎片） | 高（预分配连续布局） | 更可控的内存预算 |
| 流水重叠度 | 低（单份 buffer，必须等 MTE3 写完才能开始下一轮） | 高（双缓冲支持读写并行） | 支持 pingpong 流水 |

> 适用算子族：`softmax`（含 `softmax`、`log_softmax`、`softmax_cross_entropy` 等变体）以及内嵌 online softmax 的 `flash_attention`、`sparse_flash_attention`。

## 2. 架构概览

### 2.1 存储层级与数据流

```
GM (Global Memory)
  │
  │ MTE3 (workspace ←→ GM)
  ▼
UB Buffer [softmaxMax_ping | softmaxMax_pong | softmaxSum_ping | softmaxSum_pong | softmaxExp_ping | softmaxExp_pong]
  │
  │ Vector PIPE (SoftmaxFlashV2 / online softmax)
  ▼
UB Output → MTE3 → GM
```

### 2.2 常驻 Buffer 布局

```
UB 地址空间:
[softmaxMaxBuff: SOFTMAX_TMP_BUFFER_SIZE * preLoadNum]
[softmaxSumBuff: SOFTMAX_TMP_BUFFER_SIZE * preLoadNum]
[softmaxExpBuff: SOFTMAX_TMP_BUFFER_SIZE * preLoadNum]

索引计算:
outIdx = loop % preLoadNum
softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)
```

### 2.3 双缓冲原理

- **常驻**：`InitBuffer` 仅在算子初始化阶段调用一次，三个状态 buffer 常驻 UB 不释放。
- **双缓冲索引**：通过 `loop % preLoadNum` 轮询使用 ping/pong 两组 buffer。当 `preLoadNum = 2` 时，Vector PIPE 处理第 N 轮数据的同时，MTE3 可向第 N+1 轮 buffer 写入上一轮结果，实现读写并行。

### 2.4 事件同步模型

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `V_MTE3` | Vector 完成 → 允许 MTE3 覆写 | 状态 buffer 释放控制 |
| `MTE3_V` | MTE3 完成 → 允许 Vector 读取 | 状态 buffer 数据就绪 |

> 注：具体事件类型需根据实际 PIPE 配置调整，核心原则是「Vector 计算」与「MTE3 搬运」通过双缓冲解耦。

## 3. 关键参数配置

```cpp
// Host 侧 TilingData（或 constInfo）
struct SoftmaxConstInfo {
    uint32_t preLoadNum;           // 双缓冲深度，通常取 2
    uint32_t softmaxTmpBufferSize; // 单个状态 buffer 大小，通常 = 2K（2048 bytes）
};

// Kernel 侧 UB buffer 定义
TBuf<QuePosition::VECCALC> softmaxMaxBuff;
TBuf<QuePosition::VECCALC> softmaxSumBuff;
TBuf<QuePosition::VECCALC> softmaxExpBuff;

// InitBuffer 一次性分配（构造函数或 Init 阶段）
pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
```

### 3.1 preLoadNum 选取原则

| preLoadNum | UB 占用 | 流水重叠度 | 适用场景 |
|-----------|---------|-----------|---------|
| 1 | 3 × 2K = 6KB | 无（串行） | UB 极端紧张，仅做状态常驻不求重叠 |
| **2** | 3 × 2K × 2 = **12KB** | pingpong 双缓冲 | **默认推荐**，读写并行 |
| 3 | 18KB | 三段流水 | S2 循环极长、MTE3 延迟高 |
| 4 | 24KB | 四段流水 | UB 充裕且 MTE3 为瓶颈时考虑 |

### 3.2 SOFTMAX_TMP_BUFFER_SIZE 计算

```
SOFTMAX_TMP_BUFFER_SIZE = row_length * sizeof(COMPUTE_T)

典型值（row_length = 512, COMPUTE_T = float）:
SOFTMAX_TMP_BUFFER_SIZE = 512 * 4 = 2048 bytes
```

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

```cpp
for (uint32_t s2Loop = 0; s2Loop < s2LoopNum; s2Loop++) {
    // 每轮都重新分配 —— 开销大
    TBuf<QuePosition::VECCALC> softmaxMaxBuff;
    TBuf<QuePosition::VECCALC> softmaxSumBuff;
    TBuf<QuePosition::VECCALC> softmaxExpBuff;
    pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE);
    pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE);
    pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE);

    LocalTensor<COMPUTE_T> softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
    LocalTensor<COMPUTE_T> softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();
    LocalTensor<COMPUTE_T> softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();

    // 计算 score = Q_i * K_j^T
    // ... matmul 结果在 mmResUb ...

    // Softmax 计算
    SoftmaxFlashV2<...>(mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb,
                        softmaxExpUb, inSumTensor, inMaxTensor, ...);

    // 每轮结束 buffer 自动释放，但下一轮重新分配
}
```

### 4.2 optimized 版本（优化后）

```cpp
// 阶段 1：Init 阶段一次性分配（仅执行一次）
pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);

// 获取完整的常驻 UB tensor
LocalTensor<COMPUTE_T> softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();

for (uint32_t s2Loop = 0; s2Loop < s2LoopNum; s2Loop++) {
    // 双缓冲索引
    uint32_t outIdx = s2Loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T);

    // 计算 score = Q_i * K_j^T
    // ... matmul 结果在 mmResUb ...

    // Softmax 计算，使用常驻 buffer + 偏移索引
    SoftmaxFlashV2<...>(mmResUb, softmaxSumUb[softmaxOutOffset],
        softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, ...);

    // 可选：MTE3 将本轮结果写回 GM/workspace，与下一轮 Vector 计算重叠
    // WaitFlag<V_MTE3>(outIdx);   // 等待上一轮 MTE3 完成
    // MTE3 搬运...
    // SetFlag<MTE3_V>(outIdx);    // 通知下一轮数据就绪
}
```

### 4.3 双缓冲流水示意（preLoadNum = 2）

```
时间线 →
─────────────────────────────────────────────────────
S2 Loop 0:  [Vector: softmax on ping]  [MTE3: write ping result]
S2 Loop 1:                                [Vector: softmax on pong]  [MTE3: write pong result]
S2 Loop 2:                                                          [Vector: softmax on ping]  ...
```

> 当 Vector 处理 Loop N（pong buffer）时，MTE3 可同时写 Loop N-1（ping buffer）的结果到 GM/workspace，消除 MTE3 等待。

## 5. 从 naive 到 state_resident 的关键修改点

| 修改项 | naive（优化前） | state_resident（优化后） |
|--------|---------------|------------------------|
| buffer 分配位置 | S2 循环内，每轮 `InitBuffer` | 算子 Init 阶段，**一次性分配** |
| buffer 生命周期 | 每轮结束释放，下轮重新分配 | **常驻 UB**，循环结束不释放 |
| buffer 数量 | 单份（3 个） | `3 × preLoadNum` 份，支持双缓冲 |
| buffer 索引 | 固定偏移 0 | `loop % preLoadNum` 轮询索引 |
| 地址计算 | `buff.Get<T>()` | `buff.Get<T>()[outIdx * size / sizeof(T)]` |
| PipeBarrier 数量 | 每轮分配/释放触发多次 | 仅 Init 阶段一次，循环内无 |
| MTE3 与 Vector 重叠 | 无（串行） | 有（双缓冲解耦，可选事件同步） |
| UB 预算确定性 | 低（动态分配碎片化） | 高（预分配总量 = 3×2K×preLoadNum） |

## 6. 注意事项 / 约束

1. **UB 空间约束**：常驻占用 `3 × SOFTMAX_TMP_BUFFER_SIZE × preLoadNum`。以 `preLoadNum = 2`、`SOFTMAX_TMP_BUFFER_SIZE = 2048` 为例，总占用 12KB。需确保 UB 总预算（通常为 256KB）扣除该部分后仍有足够空间给其他 buffer（如输入 tile、输出 tile、中间计算 buffer）。

2. **preLoadNum 上限**：受 UB 总量限制。若 `preLoadNum` 过大导致 UB 溢出，编译/仿真阶段会报错。建议按以下公式校验：
   ```
   totalResident = 3 × SOFTMAX_TMP_BUFFER_SIZE × preLoadNum
   totalResident + otherBuffers ≤ UB_SIZE
   ```

3. **状态 buffer 与 Softmax API 的兼容性**：`SoftmaxFlashV2` 等高级 API 要求传入的 `softmaxMaxUb`、`softmaxSumUb`、`softmaxExpUb` 为 `LocalTensor` 类型。使用偏移索引后的子 tensor 需确保类型和形状匹配 API 要求。

4. **初始化清零**：首次使用常驻 buffer 前，建议通过 `Duplicate` 或 `DataCopy` 将初始值（如 max = -inf, sum = 0）写入，避免残留数据干扰 online softmax 的累积逻辑。

5. **与 workspace 的关系**：Flash Attention 场景下，online softmax 的 intermediate m、l、O 通常需要 GM workspace 跨 S2 tile 传递。常驻优化解决的是 **UB 内部 buffer 分配**，不改变 GM workspace 的使用逻辑。

## 7. 实施常见问题与解决方案

### 问题 1：UB 内存溢出

**现象**：编译或仿真报错，提示 UB 分配失败或溢出。

**原因**：`preLoadNum` 设置过大，或 `SOFTMAX_TMP_BUFFER_SIZE` 计算错误（如未按 `sizeof(COMPUTE_T)` 对齐），导致常驻 buffer 总占用超出 UB 容量。

**解决方案**：
- 降低 `preLoadNum`（如 3→2 或 2→1）
- 校验 `SOFTMAX_TMP_BUFFER_SIZE` 是否按 32B 或 64B 对齐（Ascend 硬件对齐要求）
- 使用 `printf` 或调试工具打印 UB 总占用：
  ```cpp
  // 调试打印
  printf("UB resident size: %u bytes\n",
         3 * SOFTMAX_TMP_BUFFER_SIZE * preLoadNum);
  ```

### 问题 2：softmax 结果错误（数值漂移）

**现象**：精度校验失败，softmax 输出与参考实现不一致。

**原因**：双缓冲索引错误，如 `softmaxOutOffset` 计算时未除以 `sizeof(COMPUTE_T)`，导致指针偏移到错误位置；或 `preLoadNum` 为 1 时仍使用 `% preLoadNum` 索引（结果始终为 0，逻辑正确但无优化效果）。

**解决方案**：
- 校验偏移计算：
  ```cpp
  // 正确：按元素个数偏移
  uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T);
  // 错误：按字节偏移直接传给 LocalTensor
  // uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE; // ❌
  ```
- 在首循环前显式初始化状态 buffer：
  ```cpp
  Duplicate(softmaxMaxBuff.Get<COMPUTE_T>(), FLOAT_NEG_INF, 3 * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) * preLoadNum);
  ```

### 问题 3：preLoadNum = 2 但无性能提升

**现象**：cycle 数与 naive 版本接近，双缓冲未生效。

**原因**：仅做了 buffer 常驻，但未在 Vector 和 MTE3 之间插入事件同步实现真正的流水重叠。`SoftmaxFlashV2` 调用后紧跟 MTE3 写回，仍是串行执行。

**解决方案**：插入 `SetFlag` / `WaitFlag` 事件对，将 MTE3 写回与下一轮 Vector 计算解耦：
```cpp
// 正确的流水同步模式
for (uint32_t s2Loop = 0; s2Loop < s2LoopNum; s2Loop++) {
    uint32_t curIdx = s2Loop % preLoadNum;
    uint32_t prevIdx = (s2Loop + preLoadNum - 1) % preLoadNum;

    // 等待上一轮 MTE3 写回完成（释放当前 buffer）
    WaitFlag<MTE3_V>(curIdx);

    // Vector 计算
    SoftmaxFlashV2<...>(...);

    // 通知 MTE3 可以写回当前 buffer
    SetFlag<V_MTE3>(curIdx);

    // MTE3 异步写回上一轮结果（与下一轮 Vector 计算重叠）
    // 注意：实际代码中 MTE3 写回应在 SetFlag 之后、下一轮 WaitFlag 之前完成
}
```

### 问题 4：与 online softmax 的 m/sum 累积逻辑冲突

**现象**：多 tile（S2 循环）场景下，softmax 结果跨 tile 不一致。

**原因**：online softmax 要求跨 tile 维护 running max 和 running sum。若每个 S2 循环使用独立的 buffer（双缓冲），但未正确将上一轮状态传递到下一轮（如通过 GM workspace），会导致状态断裂。

**解决方案**：
- 明确区分「UB 内部状态 buffer」（本优化，用于单轮 softmax 计算）和「跨 tile 累积状态」（通过 GM workspace 传递，属于 online softmax 算法本身）。
- 本优化不改变 online softmax 的跨 tile 状态传递逻辑，仅优化单 tile 内的 UB buffer 分配策略。

### 问题总结

| # | 问题 | 根因 | 解决方案 | 影响 |
|---|------|------|---------|------|
| 1 | UB 溢出 | preLoadNum 过大或 size 未对齐 | 降低 preLoadNum、校验对齐 | 编译/仿真失败 |
| 2 | 精度失败 | 偏移计算错误或 buffer 未初始化 | 校验 offset 公式、首循环清零 | 结果错误 |
| 3 | 性能未提升 | 缺少事件同步，仍是串行 | 插入 SetFlag/WaitFlag | 性能无改善 |
| 4 | 跨 tile 不一致 | 混淆 UB 优化与 online softmax 状态传递 | 保持 GM workspace 累积逻辑不变 | 精度失败 |

## 8. 实测性能、叠加关系与自检清单

### 8.1 与其他优化的叠加关系

| 优化 | 叠加可行性 | 说明 |
|------|-----------|------|
| **pingpong 双缓冲** | ✅ 高度兼容 | 本优化是 pingpong 在 softmax 状态 buffer 上的具体应用，`preLoadNum` 对应 pingpong 深度 |
| **online softmax 算法** | ✅ 必要前置 | 状态常驻优化的前提是使用 online softmax（跨 S2 循环维护状态），两者缺一不可 |
| **FP32 中间计算** | ✅ 兼容 | 状态 buffer 的 `COMPUTE_T` 可设为 float，与 FP32 数值稳定性优化共存 |
| **mte2_preload** | ⚠️ 部分兼容 | 若 softmax 前有 MTE2 预取，需确保预取 buffer 与常驻 buffer 不冲突 |
| **swat / streamk** | ❌ 不适用 | 这两种优化针对 MatMul CUBE 核心，与 softmax Vector 核心无直接交集 |

### 8.2 选型决策

```
if (算子包含 online softmax && S2 循环次数 > 1):
    → 启用 state_resident 优化
    → preLoadNum = 2（默认，UB 充裕时可试 3）
else:
    → 无需本优化（单轮 softmax 无循环复用收益）
```

### 8.3 自检清单

- [ ] `InitBuffer` 仅在算子初始化阶段调用一次，不在 S2 循环内
- [ ] `preLoadNum` 取值为 1/2/3/4，且满足 UB 预算约束
- [ ] `softmaxOutOffset` 计算正确：`outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)`
- [ ] 首循环前状态 buffer 已初始化（max = -inf, sum = 0）
- [ ] 双缓冲索引与事件同步（SetFlag/WaitFlag）配套使用，实现真正流水
- [ ] 精度校验通过（与 naive 版本对比，误差在 1e-5 以内）
- [ ] cycle 数对比：optimized / naive < 0.9（至少 10% 提升，通常 15-25%）
