---
name: ascendc-ub-budget
description: "UB 容量预算：910B/910B3/910B4 各家 UB 大小、queue/calcBuf/TBuf 的字节账本算法、BUFFER_NUM 与 TILE_LENGTH 的常见安全组合，以及 UB OOB 时的诊断路径。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC UB 容量预算

UB（Unified Buffer）是 AI Vector Core 的 on-chip 暂存。**它是有限且硬性的硬件资源**——一旦 `pipe_->InitBuffer` 申请总和超过物理 UB，编译过得了，跑起来必然 UB OOB（errno 507035 vector core exception）。

绝大多数"我把 BUFFER_NUM 从 2 抬到 3 就崩"或"我想 batch 4 行就崩"的事故，都是没算账。这个 skill 教你**写代码之前就把字节账算清楚**。

## 1. 各设备 UB 容量

| 设备 | 每个 Vector Core 的 UB | 备注 |
|---|---|---|
| 910B (Atlas 800T A2 推理卡) | **192 KB** | 单核 UB |
| 910B3 (训练 8 卡) | **192 KB** | 单核 UB |
| 910B4 (训练 8 卡，紧凑) | **128 KB** | 较小 |
| 310B / 310P | 256 KB | aiv 与 aic 不同，本表只列 aiv |

**编程时按目标设备的最小 UB 算账**。如果代码要跑 910B 也跑 910B4，按 128 KB 上限设计。

## 2. UB 字节账本公式

```text
ub_used = sum(InitBuffer 每次申请的字节数)
        + 编译器隐式 scratch  (~ 2–4 KB，留作 headroom)
```

每次 `pipe_->InitBuffer` 申请的字节数：

```text
TQue<POSITION, BUFFER_NUM>:   buffer_bytes_per_alloc × BUFFER_NUM
TBuf<POSITION>:               buffer_bytes  (BUFFER_NUM 永远 = 1，不存在多 buffer)
```

举个 elementwise unary kernel 的例子：

```cpp
pipe_->InitBuffer(inQueue_,  BUFFER_NUM, TILE_LENGTH * sizeof(T));     // VECIN
pipe_->InitBuffer(outQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));     // VECOUT
pipe_->InitBuffer(calcBuf_,  2 * TILE_LENGTH * sizeof(float));         // VECCALC

// 假设 BUFFER_NUM=2, TILE_LENGTH=4096, T=float (sizeof=4)
ub_used = 2 * (4096 * 4)      // inQueue:   32 KB
        + 2 * (4096 * 4)      // outQueue:  32 KB
        + 2 * 4096 * 4        // calcBuf:   32 KB
        = 96 KB
// 192 KB 的卡上还有 96 KB headroom，安全
```

同样 kernel，把 `BUFFER_NUM` 从 2 抬到 3，`TILE_LENGTH` 抬到 8192（half/bf16 想塞更多）：

```cpp
// BUFFER_NUM=3, TILE_LENGTH=8192, T=half (sizeof=2)
ub_used = 3 * (8192 * 2)      // inQueue:   48 KB
        + 3 * (8192 * 2)      // outQueue:  48 KB
        + 2 * 8192 * 4        // calcBuf:   64 KB
        = 160 KB
// 192 KB 卡: 还剩 32 KB 给 scratch，紧但可行
// 128 KB 卡: 已经 OOB
```

## 3. 设计阶段的预算表（910B3, 192 KB）

写新 kernel 时先估算需要哪几个 buffer，按下表挑组合：

| 用途 | 推荐 BUFFER_NUM | 单 buffer 字节数 | TILE_LENGTH 上限 |
|---|---|---|---|
| 单输入单输出 elementwise (fp32) | 2 | TILE × 4 | 8192 |
| 单输入单输出 elementwise (fp16/bf16) | 2 | TILE × 2 | 16384 |
| 单输入单输出 + 一个 fp32 calcBuf | 2 | TILE × 2 + 2·TILE × 4 | 8192 |
| 双输入单输出（如 add/mul） | 2 | 2 × (TILE × 2) | 12288 |
| Reduce-broadcast (softmax/layernorm) | 1–2 | TILE × 4 + 3·TILE × 4 | 4096 |
| Cube + Vector fused | 1 | 复杂 | 见 §6 |

**经验法则**：把 `ub_used` 控制在硬件 UB 的 80% 以下（910B3: 154 KB），留 20% 给编译器隐式 scratch、stack、tiling struct 等。

## 4. 多 buffer (BUFFER_NUM) 的真实代价

抬高 BUFFER_NUM 的动机是让 MTE2 (load) 和 V (vector compute) 重叠，理论上能 hide DMA latency。但代价是 **UB 占用按 BUFFER_NUM 倍数膨胀**。具体收益看 kernel 的 compute-to-memory ratio：

| 场景 | BUFFER_NUM=1 | BUFFER_NUM=2 | BUFFER_NUM=3 |
|---|---|---|---|
| compute >> DMA（如 exp 链长的 unary） | 浪费 UB | **最佳** | 边际收益 < 5% |
| compute ≈ DMA（如简单 add） | 慢 | **最佳** | 几乎无收益 |
| DMA >> compute（如 copy/cast）| 严重瓶颈 | 改善大 | **可能值得**，但常被 UB 大小约束掉 |

**不要默认把 BUFFER_NUM 推到 3**。先用 2 走通，profiling 看 MTE2 是不是真的没和 V 重叠，再决定。

## 5. calcBuf 与 TBuf 的取片

`calcBuf` / `TBuf` 是用户自己管理的暂存区，常切成几个 float buffer 给中间结果用。**切的时候必须用编译期常量偏移**（详见 [[ascendc-localtensor-subviews]]）：

```cpp
// calcBuf size = 3 * TILE_LENGTH * sizeof(float)
pipe_->InitBuffer(calcBuf_, 3 * TILE_LENGTH * sizeof(float));

// 在 Compute 里：
auto c = calcBuf_.Get<float>();                       // [0,             TILE_LENGTH)
auto w = calcBuf_.Get<float>()[TILE_LENGTH];          // [TILE_LENGTH,  2*TILE_LENGTH)
auto z = calcBuf_.Get<float>()[2 * TILE_LENGTH];      // [2*TILE_LENGTH,3*TILE_LENGTH)
// 三个 buffer 完全不重叠
```

**减 calcBuf 是抬 BUFFER_NUM 的常见手段**：很多 unary kernel 用了 2-3 个 float buffer，但其实可以复用（compute 完一个就覆盖它），把 3 个 buffer 压成 2 个能省 1×TILE×4 字节，进而把 BUFFER_NUM 从 2 抬到 3。这是 swi_glu / softmax 一类 reduce kernel 真正能做出 perf 的方向。

## 6. Cube + Vector 共同 kernel 的额外预算

带 Cube 的 kernel 还要扣 L0A/L0B/L0C：

| 缓冲 | 容量 (910B3) |
|---|---|
| L0A | 64 KB |
| L0B | 64 KB |
| L0C | 128 KB |
| L1  | 512 KB |

Vector 部分仍走 UB，但要注意 Cube 完成后到 Vector 之间用 `SetFlag<HardEvent::M_V>` / `WaitFlag<HardEvent::M_V>` 配对。Cube 写到 L0C 的结果**不会自动进 UB**，需要 fixpipe 或 DataCopy 显式搬。

## 7. 优化时的 UB 账本模板

性能优化常会同时做三件事：调大 tile、增加 queue depth、增加 calcBuf。三者不能独立看，必须在同一张账本里评估。

```text
输入队列:    input_count  × BUFFER_NUM × TILE × sizeof(input_dtype)
输出队列:    output_count × BUFFER_NUM × TILE × sizeof(output_dtype)
fp32 scratch: scratch_f32 × TILE × 4
half scratch: scratch_h16 × TILE × 2
reduce tmp:   reduce_tmp_bytes
常量/小表:    const_bytes
headroom:     至少 20% UB
```

示例：一个 fused activation，fp16 输入输出，内部需要 3 段 fp32 scratch：

```text
BUFFER_NUM=2, TILE=4096:
in/out queue = 2 × 2 × 4096 × 2 = 32 KB
fp32 scratch = 3 × 4096 × 4 = 48 KB
total        = 80 KB  // 安全

BUFFER_NUM=3, TILE=8192:
in/out queue = 2 × 3 × 8192 × 2 = 96 KB
fp32 scratch = 3 × 8192 × 4 = 96 KB
total        = 192 KB // 910B3 已无 headroom，实际高风险
```

如果想把 `BUFFER_NUM` 从 2 提到 3，常见做法不是硬抬，而是先压缩 live range：

```cpp
// 差：三个 scratch 同时常驻。
auto a = calc.Get<float>();
auto b = calc.Get<float>()[TILE];
auto c = calc.Get<float>()[2 * TILE];

// 好：确认 a 的结果不再使用后，让 c 复用 a 的空间。
auto a = calc.Get<float>();
auto b = calc.Get<float>()[TILE];
auto c = a;
```

per-dtype tile 应独立算账。fp32 path 可能没有 Cast scratch，tile 可以更大；fp16/bf16 path 虽然输入更小，但常需要 fp32 scratch，未必能翻倍。

```cpp
if (dtype == DTYPE_FLOAT) {
  tile = 8192;   // 无 fp32 cast scratch
} else {
  tile = 6144;   // 输入小，但 scratch 多
}
```

多行合批也要算账：`rowsPerTile × rowLength × dtypeSize` 必须同时容纳输入、输出、临时规约和 padding。若合批导致 UB 超 80%，优先减少合批行数，而不是砍掉必要的精度 buffer。

## 8. UB OOB 实战诊断

`errno 507035 vector core exception` + `errorStr: VEC instruction error: the ub address out of bounds` 出现时：

1. **先算账本**：把当前 kernel 的所有 `InitBuffer` 字节加起来。超过 80% UB 容量基本必崩。
2. **检查 §3 表里的组合**：你是不是把 `BUFFER_NUM=3` 配上了 `TILE_LENGTH=8192 + fp16` + 多个 calcBuf？
3. **检查子视图偏移**：见 [[ascendc-localtensor-subviews]]，runtime 偏移子视图也会以同样的 errno 报。
4. **加 print 不行**：device 端 `printf` 在多核下不稳定。改成把可疑 LocalTensor 的前几个元素 `DataCopy` 回 GMEM，host 端 dump。
5. **二分定位**：注释掉一半 compute 看是否还崩；逐步加回来定位到具体那条 intrinsic。

## 9. 不要做的事

- 不要"先把所有 buffer 开到最大，跑通了再压缩" —— 跑不通就回不来，应当从最小 BUFFER_NUM=1、小 TILE 起步逐步抬。
- 不要在 kernel 里调 `aclrtMalloc` —— UB 是静态分配的，运行时分配会破坏 pipe sync 调度。
- 不要假设"编译过 = UB 没超" —— 编译器只检查 `InitBuffer` 的 sum 是否超物理 UB（且对子视图偏移和 tail 路径几乎不查），实际 OOB 经常运行时才报。
