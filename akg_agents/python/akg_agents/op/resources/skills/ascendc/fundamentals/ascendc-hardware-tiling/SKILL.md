---
name: ascendc-hardware-tiling
description: "AscendC 硬件与 tiling 基础：UB/L1/L0 容量、Vector/Cube 核关系、DataCopy 对齐、blockDim/tail 计算和批量算子开发的 tiling 不变量。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC 硬件与 Tiling 基础

在编写或修改 AscendC kernel 前使用本 skill，尤其适用于一个算子需要覆盖多组 shape，或多个算子需要批量复用同一套 tiling 骨架的场景。

## 1. 硬件容量与查询原则

host tiling 代码不要散落硬编码芯片参数。SDK 能查询时优先查询；若当前 seed 已经提供常量，也应集中到一个 helper 中。

| 缓冲区 | 典型用途 | 开发规则 |
|---|---|---|
| UB | Vector 输入、输出、临时张量 | tile 设计必须容纳所有 live LocalTensor 和队列 buffer |
| L1 | Cube 输入复用或较大 staging | 只有存在真实复用时使用 |
| L0A/L0B | Cube 左/右操作数 | 按 Cube API 约束设计，避免超过 64KB 级容量 |
| L0C | Cube 累加结果 | 显式控制累加 shape 和 dtype |

A2/A3/A5 类设备上 VectorCore 与 CubeCore 可能分离。若一个 kernel 同时包含 Cube 和 Vector 工作，应保证两类核心的任务划分不会互相长时间等待。

## 2. Tiling Data 契约

tiling struct 应小、稳定、host/kernel 完全一致：

```cpp
struct TilingData {
    uint32_t total;
    uint32_t tile;
    uint32_t tail;
    uint32_t tilesPerCore;
    uint32_t coreNum;
};
```

规则：

- 使用固定宽度整数类型，例如 `uint32_t`、`int64_t`。
- host 与 kernel 中字段顺序、类型、含义必须一致。
- 字段名明确区分元素数和字节数，例如 `elemLen` 与 `byteLen`。
- `DataCopy` 长度单位必须和 API 要求一致，不要混用元素数和字节数。
- 将可预计算的 scalar 字段留在 host；过大的 tiling struct 会增加 setup 开销。

多模式 kernel 推荐显式加入 mode 字段，而不是在 device 端反复解析 shape：

```cpp
enum Mode : uint32_t {
    MODE_GENERIC = 0,
    MODE_SAME_SHAPE = 1,
    MODE_LAST_DIM_BROADCAST = 2,
    MODE_SMALL_REDUCE = 3,
};

struct TilingData {
    uint32_t mode;
    uint32_t total;
    uint32_t inner;
    uint32_t tile;
    uint32_t tilesPerCore;
    uint32_t coreNum;
};
```

规则：

- mode 必须来自 host 侧的 dtype、rank、stride、broadcast、axis、numel 等语义判断。
- mode 只表示算法路径，不要把完整 shape 编码成一堆 mode。
- generic mode 必须保留完整语义覆盖。
- device 端 mode 分支应在外层选择，避免在每个元素内反复判断。

## 3. Block 与 Tail 公式

一维可展平任务的安全起点：

```text
core_num       = min(max_cores, ceil_div(total, min_work_per_core))
work_per_core  = ceil_div(total, core_num)
core_start     = block_idx * work_per_core
core_len       = min(work_per_core, total - core_start)
tile_count     = core_len / tile_len
tail_len       = core_len % tile_len
```

kernel 侧必须满足：

- 若 `core_start >= total`，在使用 GM 指针前直接返回。
- 最后一个有效 core 使用 `core_len`，不要使用名义上的 `work_per_core`。
- `tail_len == 0` 时不要执行零长度或旧长度 tail copy。
- tail vector mask 的有效长度等于真实剩余元素数。

## 4. 对齐规则

- GM/UB 搬运优先使用 32B 对齐的 tile size。
- 非 32B 对齐数据使用 `DataCopyPad` 或单独 tail 路径。
- profiling 出现 UB bank conflict 时，让热点 LocalTensor 偏移至少错开 32B。
- vector API 的 repeat stride、block stride 单位需要逐项确认，不要直接套用 host 字节数。

## 5. Queue 与 Pipeline 不变量

队列式 kernel 保持如下结构：

```cpp
auto in = inQueue.AllocTensor<T>();
DataCopy(in, gmIn + offset, len);
inQueue.EnQue(in);

auto ready = inQueue.DeQue<T>();
Compute(ready, outLocal, len);
inQueue.FreeTensor(ready);
```

检查项：

- 每个 `AllocTensor` 都有一个 `FreeTensor`。
- 每个 `EnQue` 都有一个匹配的 `DeQue`。
- 分支不能在 `AllocTensor` 后跳过释放。
- tail 分支不能跳过主路径后续需要的 `DeQue`。
- 只有存在明确数据依赖时才加 `PipeBarrier`；过多 barrier 会破坏 overlap。

## 6. 批量开发规则

处理多个算子时：

1. 先将算子归类为 elementwise、broadcast、reduction、indexed、matmul-like 或 fused。
2. 每一类只维护一个 tiling 骨架。
3. 每个算子只改语义点：输出 shape、dtype 分支、计算表达式、归约轴、tail 策略。
4. direct-invoke wrapper 和 CMake 布局保持稳定。
5. 不通过修改 tolerance 或参考语义让 kernel 通过。

## 7. Tiling 与性能分桶

当同一算子覆盖多组 shape 时，tiling 需要同时服务正确性和性能。推荐最少记录以下分桶字段：

```text
same_shape:
last_dim_broadcast:
small_reduce:
single_tile_row:
large_integral:
aligned_main:
tail_only_pad:
```

示例 host tiling：

```cpp
bool sameShape = x.sizes() == y.sizes();
bool bothContig = x.is_contiguous() && y.is_contiguous();
bool lastDimBroadcast = x.dim() >= 2 && y.dim() == 1 &&
                        y.size(0) == x.size(x.dim() - 1);

if (sameShape && bothContig) {
    tiling.mode = MODE_SAME_SHAPE;
    tiling.total = x.numel();
} else if (lastDimBroadcast && bothContig) {
    tiling.mode = MODE_LAST_DIM_BROADCAST;
    tiling.inner = y.size(0);
    tiling.total = x.numel();
} else {
    tiling.mode = MODE_GENERIC;
}
```

示例 device 分支：

```cpp
if (mode_ == MODE_SAME_SHAPE) {
    ProcessFlatten();
} else if (mode_ == MODE_LAST_DIM_BROADCAST) {
    ProcessRowBroadcast();
} else {
    ProcessGenericIndexMapping();
}
```

不要让 generic index mapping 成为所有输入的默认性能路径；它通常包含 `div/mod` 和多级 stride 计算，只适合作为 fallback。
