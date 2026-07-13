---
name: ascendc-op-patterns
description: "AscendC 常见算子家族实现模板：elementwise、broadcast、reduction、softmax-like、index/gather 和 matmul epilogue。用于批跑时快速选择初始 kernel 结构。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "elementwise,broadcast,reduction,indexed,matmul-like,fused"
---

# AscendC 常见算子模板

在修改 kernel 数学逻辑前使用本 skill 选择实现骨架。推荐先得到正确的 direct-invoke kernel，再用 profiling 文档做性能优化。

## 1. 模板选择表

| 算子形态 | 初始骨架 | 主要风险 |
|---|---|---|
| unary/binary elementwise | flatten + vector tile | dtype cast 与 tail mask |
| broadcast elementwise | 线性输出 index 映射到输入 offset | 整数索引开销 |
| row reduction | 每核处理一行或若干行 | 累加精度 |
| large reduction | 分段归约 + 二阶段合并 | workspace 或 atomic 成本 |
| softmax/logsumexp | max pass + exp/sum pass + normalize | 溢出和 workspace |
| gather/scatter/index | contiguous 快路径 + generic 路径 | 越界和写冲突 |
| matmul + epilogue | 保持 Cube 主路径，尽量融合 UB/L0C epilogue | Cube/Vector 负载平衡 |

## 2. Elementwise 骨架

输出 contiguous 时优先展平成一维元素数：

```text
for each core:
  CopyIn each input tile
  Compute vector expression
  CopyOut output tile
  handle tail
```

规则：

- tile 长度同时满足 vector block size 和 32B 搬运粒度。
- scalar 参数的预处理放到 host tiling。
- 输出 dtype 与计算 dtype 不同时，只在最后统一 Cast。
- `where` 类算子的 predicate 在 UB 内生成，避免中间结果写回 GM。

## 3. Broadcast 骨架

为常见 broadcast 拆快路径：

- same shape：直接 elementwise。
- trailing-dim broadcast：沿最后一维向量化。
- scalar input：每个 tile 只加载一次 scalar。
- general broadcast：使用 index mapping fallback。

不要让所有 shape 都走 general index mapping；它通常会变成 scalar-bound。

## 4. Reduction 骨架

编码前先定义：

```text
outer  = product(dims before reduced axis)
reduce = product(reduced dims)
inner  = product(dims after reduced axis)
```

选择顺序：

1. `inner == 1` 时优先做连续块归约。
2. `reduce` 较小时，一个 core 处理一行或多行。
3. `reduce` 很大时，拆成多 core 分段归约，再用 workspace 或 atomic 合并。
4. fp16/bf16 的敏感归约按参考精度要求使用 fp32 accumulator。

tail 规则：

- 最后一个 reduction tile 必须 mask 无效 lane。
- max/min 使用正确 identity value。
- sum/prod 显式初始化，不依赖 UB 默认内容。

## 5. Softmax-Like 骨架

数值稳定路径：

```text
row_max = reduce_max(x)
tmp = exp(x - row_max)
row_sum = reduce_sum(tmp)
y = tmp / row_sum
```

规则：

- row 能放入 UB 时，优先保留中间结果在 UB。
- 长 row 使用多 pass tiling 和 workspace。
- epsilon 只能在参考语义允许时引入。
- 输出 shape 和 dtype 必须与参考实现一致。

## 6. Indexed 算子

适用于 gather、scatter、nonzero、index-put 等：

- 可行时在 host 侧验证 index dtype 和 bounds。
- 将 contiguous 快路径与 generic 路径分开。
- 写冲突必须显式定义：atomic/add/last-write 等语义不能混淆。
- 不要把 indexed 语义静默改写成 dense elementwise。

## 7. Matmul Epilogue

若任务是 matmul-like 加 bias、activation、scale：

- 先保持 Cube tiling 主路径正确。
- 简单 epilogue 尽量在写回 GM 前融合。
- 参考语义要求时使用 fp32 accumulation。
- quant/dequant scale 的 layout 写入 tiling data，不用隐式假设。

## 8. 从样本形态选择快路径

多 shape 算子不要只按算子名选一个通用骨架。先把输入样本归入少数语义模式，再为高频或高耗时模式拆快路径，最后保留完整 generic fallback。

常见分桶：

| 分桶 | 判定条件 | 推荐骨架 |
|---|---|---|
| same-shape contiguous elementwise | 所有输入 shape 相同且 contiguous | flatten + bulk DataCopy + vector tile |
| scalar broadcast | 某输入 numel=1 | 每 tile 只加载一次 scalar 或 host 侧传 scalar |
| last-dim broadcast | `(outer, D)` 与 `(D,)` 或高维 trailing broadcast | 按行处理，复用 broadcast 输入 |
| small-row reduction | reduce dim 很小、row 数很多 | 多行合批，必要时标量规约 |
| single-tile row reduction | 单行能完整放入 UB | 一次 CopyIn，UB 内完成所有 pass |
| large-row reduction | 单行超过 UB | 分 tile 累加 + 二阶段合并或 workspace |
| contiguous indexed segment | index 形成连续段 | DataCopy 连续块，避免逐元素 GetValue |
| identity / single segment | scatter/segment reduce 的特殊语义 | 直接 copy、sum 或局部 accumulation |
| special value | all-zero、all-NaN、constant 输入 | 语义快路径填充或跳过计算 |

host 侧推荐生成 mode 字段，kernel 侧只做轻量分支：

```cpp
enum class PatternMode : int32_t {
  kGeneric = 0,
  kSameShape = 1,
  kLastDimBroadcast = 2,
  kSmallReduction = 3,
  kSingleTileReduction = 4,
};

PatternMode Classify(const at::Tensor& x, const at::Tensor& y, int64_t axis) {
  if (x.sizes() == y.sizes() && x.is_contiguous() && y.is_contiguous()) {
    return PatternMode::kSameShape;
  }
  if (x.dim() >= 2 && y.dim() == 1 &&
      y.size(0) == x.size(x.dim() - 1)) {
    return PatternMode::kLastDimBroadcast;
  }
  if (axis == x.dim() - 1 && x.size(axis) <= 32) {
    return PatternMode::kSmallReduction;
  }
  return PatternMode::kGeneric;
}
```

device 端不要把所有模式揉进一条复杂内层循环。模式分支应尽量在 `Process` 外层决定：

```cpp
void Process() {
  if (mode_ == kSameShape) {
    ProcessSameShape();
  } else if (mode_ == kLastDimBroadcast) {
    ProcessLastDimBroadcast();
  } else {
    ProcessGeneric();
  }
}
```

## 9. 家族级实现细节

### 9.1 Elementwise / Activation

- 去掉恒等 `Adds(x, 0)` 和不必要的 fp32 往返。
- 只在参考精度要求时升精度；fp16/bf16 native 路径要单独验证误差。
- exp、log、tanh、rsqrt 链路优先融合中间阶段，避免每步都占一个 calc buffer。
- 对 all-zero、all-NaN、constant 输入可以有语义快路径，但不能改变普通输入的 dtype 覆盖。

### 9.2 Broadcast Elementwise

- same-shape、scalar、last-dim broadcast 必须先于 general index mapping。
- general broadcast 中 `div/mod` 应尽量在 host 侧合轴，kernel 内用 stride 和增量 offset。
- 若 broadcast 输入很小，可以整段加载到 UB，在一个 tile 内反复复用。

### 9.3 Reduction / Softmax-like

- small D：多行合批，或标量规约减少同步。
- medium D：一行进 UB，max/sum/normalize 在 UB 内串起来。
- large D：分 tile 累加，保存 partial，再做二阶段合并。
- softmax/logsumexp 要保持数值稳定：`max -> exp(x-max) -> sum -> normalize`，不要为了少一 pass 破坏溢出保护。

### 9.4 Indexed / Geometry

- 优先识别连续段、dim0/dim1 常见轴、identity reduce、single segment。
- 把 `GetValue/SetValue` 改成 UB 暂存 + 连续 DataCopy，是 gather/scatter/resize 类算子的首要优化方向。
- 几何类算子的 row base、plane base、weight table base 在 batch 开头预计算，内层只做增量推进。
