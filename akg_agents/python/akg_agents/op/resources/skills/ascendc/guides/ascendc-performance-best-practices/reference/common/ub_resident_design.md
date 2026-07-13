# UB/TBuf 常驻复用与 Bank 冲突规避优化设计

## 1. 优化目标

在 Vector 算子中，小尺寸参数（weight、gamma、scale）或跨迭代状态（累加器、递推结果）若每个 tile/loop 都重复从 GM 搬运，会造成大量冗余 MTE2 开销。本优化通过将这类数据常驻 UB/TBuf，在循环外一次初始化、循环内片上读写，**消除重复搬运**。同时，通过地址分配优化和计算逻辑调整**规避 UB bank 冲突**，避免 Vector 计算性能退化。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| 参数搬运次数 | 每 tile/loop 1 次 | Process 生命周期仅 1 次 | 搬运次数降低 N 倍（N=tile/loop 数） |
| 临时 buffer 总量 | `sum(stage_buffers)` | `max(concurrent_buffers)` | UB 占用降低 30-60% |
| Vector 单 Repeat 周期 | 多拍（bank 冲突，因芯片而异） | 1-2 拍（无冲突） | 计算速度提升数倍 |
| 代码复杂度 | 低 | 中（需生命周期管理） | 常驻 buffer + 分区 offset |

## 2. 架构概览

### 2.1 常驻复用 vs 分时复用

| 策略类型 | 数据生命周期 | 复用方式 | 典型场景 |
|---------|------------|---------|---------|
| **常驻复用** | 跨整个 Process 或多层循环 | 循环外搬入，循环内只读/读写 | weight、gamma、scale 小参数 |
| **分时复用** | 单次 Process 内多阶段串行 | 前一阶段结束，后一阶段覆盖 | RmsNorm→RoPE、SoftMax→后处理 |

### 2.2 UB Bank 结构与冲突原理

以 Ascend 910B 为例：UB (192KB) = 48 bank × 4KB = 16 bank group × 3 bank/group。不同芯片的 bank 数可能不同，需查阅对应硬件手册。

#### 2.2.1 Bank 冲突的硬件原理

Vector 计算单元在一个 Repeat 周期内会并行发出多个 DataBlock 的读写请求。理想情况下，这些请求被分发到不同的 bank/bank group 上并发执行，1 个 Repeat 周期完成全部访问。当多个请求因地址映射关系落到**同一 bank 或同一 bank group** 时，硬件仲裁逻辑会将其串行化，导致单 Repeat 从 1-2 拍退化到多拍（具体拍数因芯片和冲突 DataBlock 数量而异）。

> **注意**：不同芯片的 bank 结构和冲突条件不同，必须查阅对应硬件文档。下文以 NPU 架构版本 220x（Atlas A2/A3 系列）和 Atlas 350 为例说明。

#### 2.2.2 不同芯片的 UB Bank 结构

| 芯片 | UB 大小 | Bank 数 | Bank Group 数 | 每组 Bank 数 | 每 Bank 大小 | 每行大小 |
|------|---------|---------|---------------|-------------|-------------|---------|
| DAV_2201 | 192 KB | 48 | 16 | 3 | 4 KB (128 行) | 32 B |
| DAV_3510 | 256 KB | 16 | 8 | 2 | 16 KB (512 行) | 32 B |

**Bank Group 组成方式**：group 内各 bank 的编号满足 `bank_id % num_groups` 相同。以 220x 为例：
- group 0: bank 0, 16, 32
- group 1: bank 1, 17, 33
- ...
- group 15: bank 15, 31, 47

**冲突条件差异**（关键区别）：

| 冲突类型 | NPU 220x | Atlas 350 |
|---------|----------|-----------|
| 读写冲突 | 同一 **bank** | 同一 **bank** |
| 写写冲突 | 同一 **bank group** | 同一 **bank group** |
| 读读冲突 | 同一 **bank group** | 同一 **bank**（两个读操作），或**两个以上**读操作同一 bank group |

> Atlas 350 的 bank group 上有两组读口和写口，因此**两次读操作访问同一 bank group 的不同 bank 时不会冲突**，但 220x 上同一 bank group 内的任意多次并发访问都会冲突。

#### 2.2.3 DataBlock 与 Bank 映射

Vector 指令处理的数据被切分为固定大小的 **DataBlock**，每行长度为 32B。一个 Vector 指令在一个 Repeat 内最多处理 **8 个 DataBlock**（block0~block7）。各 DataBlock 根据起始地址映射到某个 bank，进而归属于某个 bank group。

**映射的核心规律**（由官方示例推导，具体公式以芯片手册为准）：
- 地址上相邻的 32B 行通常映射到相邻的 bank（按 bank 总数循环）。
- 因此，**DataBlock 之间的地址差（由 `blk_stride` 决定）直接决定了它们是否落入同一 bank group**。

以 220x 为例：
- `blk_stride = 16`：相邻 DataBlock 地址差 = 16 × 32B = 512B。bank 编号差 = 16（假设按 32B 行轮询），16 % 16 = 0，即**所有 DataBlock 落入同一 bank group**，8 拍完成一个 Repeat。
- `blk_stride = 8`：bank 编号差 = 8，block0 和 block2 的 bank 差 = 16，16 % 16 = 0，落入同一 group，4 拍完成一个 Repeat。

#### 2.2.4 冲突判断方法

**步骤 1：识别并发访问的数据源**

一个 Vector 指令（如 `Add(dst, src0, src1)`）在同一 Repeat 内会同时读取 `src0`、`src1` 并写入 `dst`。这三个操作数的 DataBlock 是并发访问的候选冲突源。

**步骤 2：分析地址间隔规律**

- **多操作数之间**：若 `dst`、`src0`、`src1` 的 buffer 起始地址在 UB 内连续分配，且间隔恰好是 bank size 的整数倍，则它们的对应 DataBlock 会周期性地映射到同一 bank/bank group。
- **单操作数内部**：若 `blk_stride` 使 DataBlock 之间的 bank 差等于 group 数（220x: 16，350: 8）或其倍数，则多个 DataBlock 会回卷到同一 bank group。

**步骤 3：通过官方工具或实验确认**

- **msProf 资源冲突占比分析**：官方提供的 msProf 工具可以采集资源冲突占比数据，直接定位 bank 冲突。具体使用方法参考《算子开发工具》文档。
- **Profiling 验证**：如果 Vector 计算耗时远高于理论值，且 MTE2 不是瓶颈，则极可能存在 bank 冲突。
- **实验验证**：对疑似冲突的代码添加 256B padding 或调整 `blk_stride` 后重新 profiling，观察周期数是否下降。

#### 2.2.5 常见冲突场景与规避原理

以下场景来自官方文档的典型示例。

**场景 1：读写冲突 —— src 与 dst 落在同一 bank**

```cpp
// 假设 x 起始地址在 bank0，y 起始地址也在 bank0（地址差为 bank size 的整数倍）
Add(dst, src, src2, ...);  // src 读 bank0，dst 写 bank0，读写冲突
```

**规避**：确保 src 和 dst 的起始地址不在同一 bank。对于 220x，通常给中间 buffer 增加 padding 使地址错开至少一个 bank。

**场景 2：写写冲突 —— dst 的多个 DataBlock 落入同一 bank group**

```cpp
// 220x 上 blk_stride=16：8 个 DataBlock 全部落入同一 bank group，8 拍完成
Adds(dst, src, scalar, MASK, 1, {1, 16, 1, 16});

// 220x 上 blk_stride=8：block0 和 block2 落入同一 bank group，4 拍完成
Adds(dst, src, scalar, MASK, 1, {1, 8, 1, 8});
```

**规避**：改为 `blk_stride = 1`（连续读），通过 `dst_gap/src_gap` 控制跨 Repeat 的地址增量。

**场景 3：读读冲突 —— 双 src 落入同一 bank group（220x），或同一 bank（350）**

```cpp
// 220x：x 和 y 起始地址差为 bank size 整数倍，DataBlock 0 同时读同一 bank group
Add(zLocal, xLocal, yLocal, ...);
```

**规避**：在 `xBuf` 后增加 256B padding，打破地址周期性，使 `x` 和 `y` 的 DataBlock 分散到不同 bank group。对于 350，还需确保两个 src 不在同一 bank。

**场景 4：连续分配多个 buffer，地址周期性重叠**

```cpp
// 220x 原始实现：x/y/z 连续分配，起始地址差为 0x4000（16KB）
// x: bank0, y: bank0（地址差 16KB = 4×bank size），z: bank0
// 一个 Repeat 内 x 和 y 同时读同一 group，x/y 和 z 同时读写同一 bank
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

**规避（220x 官方推荐）**：
```cpp
// x 多申请 256B，避免一个 Repeat 内 x 和 y 同时读同一 bank group
// y 多申请空间，确保 z 不会和 x/y 落入同一个 bank
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256);
pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

### 2.3 数据流 — 常驻复用

常驻复用：小尺寸参数（weight、gamma、scale）在 Process 入口通过 MTE2 一次搬入 UB/TBuf，后续整个生命周期内循环只读，全程不再访问 GM。

### 2.4 数据流 — 分时复用（Zone Reuse）

分时复用：单块 TBuf 按阶段划分为多个 zone。前一阶段（如 RmsNorm）使用 zone0/1/2 计算完成后数据失效，后一阶段（如 RoPE）直接复用同一物理空间的 zone0/1/2，无需额外分配 buffer。UB 临时 buffer 总量从 `sum(stage_buffers)` 降为 `max(concurrent_buffers)`。

## 3. 关键参数配置

```cpp
// 常驻复用参数
struct ResidentBufferConfig {
    uint32_t paramSize;       // 常驻参数元素数（如 hidden_size）
    uint32_t computePrecision; // 计算精度：FP16=2, FP32=4 bytes
};

// 分时复用参数
struct ZoneReuseConfig {
    uint32_t zone0Size;       // 阶段 1 临时 buffer 大小
    uint32_t zone1Size;       // 阶段 2 临时 buffer 大小
    uint32_t zone2Size;       // 阶段 3 临时 buffer 大小
};

// Bank 冲突规避参数
struct BankConflictConfig {
    uint32_t dataSize;        // 数据大小（字节）
    uint32_t paddingSize;     // Padding 大小（通常为 256B）
};
```

### 3.1 参数选取原则

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `paramSize` | 64 / 128 / 256 / 512 | head_dim 或 hidden_size |
| `paddingSize` | 256 | 使相邻 buffer 错开 bank group |
| `zoneOffset` | `rows * headSize` | 按数据量对齐划分 zone |

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

**参数每 tile 重复搬运：**
```cpp
for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
    for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
        // 每次 loop 都从 GM 搬运 weight
        LocalTensor<half> weightLocal = inQueueW.AllocTensor<half>();
        DataCopyPad(weightLocal, weightGm, copyParams);
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
        Compute(xLocalFp32, weightFp32, ...);
        inQueueW.FreeTensor(weightLocal);
    }
}
```

**分时复用反例 — 独立分配：**
```cpp
// 各阶段独立 buffer，总量 = sum
pipe.InitBuffer(rmsBuf0, 1, zoneSize);
pipe.InitBuffer(rmsBuf1, 1, zoneSize);
pipe.InitBuffer(rmsBuf2, 1, zoneSize);
pipe.InitBuffer(ropeCosBuf, 1, zoneSize);
pipe.InitBuffer(ropeSinBuf, 1, zoneSize);
// 总量 = 5 × zoneSize
```

**Bank 冲突反例 — 连续分配（220x）：**
```cpp
// 问题：x/y/z 连续分配，地址差为 dataSize。
// 若 dataSize 是 bank size（4KB）整数倍，x/y/z 的对应 DataBlock 周期性地映射到
// 同一 bank group。Add 指令同时读 x、读 y、写 z，产生严重的读写/读读冲突。
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));   // addr = 0x0
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));   // addr = 0x4000
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));  // addr = 0x8000

// 问题：220x 上 blk_stride=16，8 个 DataBlock 全部落入同一 bank group（bank 差=16，
// 16 % 16 = 0），8 拍完成一个 Repeat。
Adds(dst, src, scalar, MASK, 1, {1, 16, 1, 16});  // 全冲突！

// 问题：220x 上 blk_stride=8，block0 和 block2 落入同一 group（bank 差=16），4 拍。
Adds(dst, src, scalar, MASK, 1, {1, 8, 1, 8});    // 部分冲突！
```

### 4.2 optimized 版本（优化后）

**常驻复用 — Process 入口搬入，循环内只读：**
```cpp
__aicore__ inline void Process() {
    // 阶段 0：一次搬入 + Cast，常驻 UB
    LocalTensor<float> weightFp32 = this->inQueueW.AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);

    // 阶段 1~N：循环内只读常驻 weightFp32
    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

**分时复用 — Zone 划分：**
```cpp
// 单块 TBuf 划分为多 zone
int64_t xLocalFp32Offset = 0;
int64_t xSquareLocalOffset = rows * headSize;
int64_t xSumLocalOffset = rows * headSize * 2;

LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

// 阶段 1：使用 zone0/1/2
RmsNorm(xLocalFp32, xSquareLocal, xSumLocal, ...);

// 阶段 2：复用 zone0/1/2 做 RoPE
LocalTensor<float> ropeCosLocal = wsLocal[xLocalFp32Offset];
LocalTensor<float> ropeSinLocal = wsLocal[xSquareLocalOffset];
RoPE(ropeCosLocal, ropeSinLocal, ...);
```

**Bank 冲突规避 — Padding 错开（220x 官方推荐）：**
```cpp
// 原理：x 多申请 256B，使 x 和 y 的 DataBlock 错开 bank group；
// y 再补齐到 64KB（16 个 bank）边界，确保 z 不与 x/y 同 bank。
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256);
pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256));
pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));
```

**Bank 冲突规避 — 连续读跳写：**
```cpp
// 原理：blk_stride=1 使同一操作数内的 8 个 DataBlock 顺序映射到相邻 bank，
// 避免落入同一 bank group。dst_gap/src_gap 控制跨 Repeat 的地址增量。
// 220x 上 mask=128（8 DataBlock）时，连续读无自冲突；跳写使 dst 分散。
UnaryRepeatParams params;
params.dstBlkStride = 8;
params.srcBlkStride = 1;
Adds(dstLocal, srcLocal, 0, 128, 2, params);  // 连续读，跳写
```

## 5. 从 naive 到 ub_resident 的关键修改点

| 修改项 | naive（优化前） | ub_resident（优化后） |
|--------|---------------|---------------------|
| 参数搬运 | 每 tile/loop 重复 MTE2 | Process 入口一次搬入，常驻 UB |
| 临时 buffer 总量 | `sum(stage_buffers)`（独立分配） | `max(concurrent_buffers)`（分区复用） |
| UB 地址分配 | 连续分配，无 padding | 256B padding 错开 bank group |
| Vector stride | 跳读连续写（blk_stride=16） | 连续读跳写（blk_stride=1） |
| 数据生命周期 | 每阶段独立 buffer | 串行阶段复用同一物理空间 |

## 6. 注意事项 / 约束

1. **常驻 buffer 压缩主数据 tile 空间**：常驻 buffer 会长期占用 UB，需确保剩余空间仍可容纳主数据 tile、双缓冲和临时计算空间。

2. **TBuf/VECCALC 不受队列同步保护**：需要手动使用 PipeBarrier 或明确阶段边界保证一致性。

3. **分时复用需严格串行边界**：各阶段必须具备严格的串行边界，错误判断生命周期会导致后续阶段覆盖仍在使用的数据。

4. **Bank 冲突规避需区分芯片型号**：
   - NPU 220x（A2/A3）：192KB = 48 bank × 4KB = 16 group × 3 bank，读写/写写/读读冲突条件见 2.2.2 节。
   - Atlas 350：256KB = 16 bank × 16KB = 8 group × 2 bank，两次读操作同一 group 不同 bank 不冲突。
   - 具体规格和推荐配置务必查阅对应版本的《Ascend C 算子开发最佳实践》。

5. **Padding 增加 UB 占用**：256B padding 虽然小，但多个 buffer 累积后需计入总 UB 预算。

6. **后续阶段重叠流水时需重算 zone**：若后续优化把串行阶段改造成重叠流水，原有 zone reuse 方案可能失效。

## 7. 实施常见问题与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| 常驻 buffer 后 UB 溢出 | 常驻 buffer + 主 tile + 双缓冲超预算 | 减小 tile_size 或常驻 buffer 大小；精确计算 UB 预算 |
| 分时复用数据被覆盖 | 阶段边界判断错误 | 确保前一阶段完全结束后再覆盖；使用 PipeBarrier 明确边界 |
| Vector 性能仍低 | bank 冲突未解决 | 使用 msProf 采集资源冲突占比确认；按 2.2.5 节场景逐一排查（读写/写写/读读冲突）；确认 padding 后各 buffer 的 DataBlock 不在同一 bank/bank group；检查 `blk_stride` 是否导致 8 个 DataBlock 回卷到同一 group |
| 累加器精度不足 | FP16 累加溢出 | 常驻累加器使用 FP32 精度 |

## 8. 选型决策与自检清单

### 8.1 选型决策

```
if (算子包含小尺寸参数 weight/gamma/scale 且跨 loop 不变):
    → 启用常驻复用：Process 入口搬入，循环内只读
elif (算子包含多阶段串行计算，临时 buffer 生命周期不重叠):
    → 启用分时复用：单块 TBuf 分区，阶段间覆盖
elif (Profiling 显示 Vector 计算时间异常高):
    → 启用 bank 冲突规避：padding 错开 或 连续读跳写
else:
    → 标准分配即可
```

### 8.2 自检清单

- [ ] 常驻 buffer 大小 + 主 tile + 双缓冲 ≤ UB 容量
- [ ] 分时复用各阶段有严格的串行边界（PipeBarrier）
- [ ] Bank 冲突规避：相邻 buffer 间隔 ≥ 256B 或 不在同一 bank group
- [ ] Vector API 的 blk_stride 避免导致多 DataBlock 落入同一 bank group
- [ ] 累加器/递推状态使用 FP32 精度
- [ ] 验证通过：与 naive 实现对比，结果一致