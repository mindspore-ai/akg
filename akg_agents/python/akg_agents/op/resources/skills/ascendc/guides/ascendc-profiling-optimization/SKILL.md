---
name: ascendc-profiling-optimization
description: "AscendC profiling 到优化动作的决策表：VEC/MTE/CUBE/SCALAR bound、bank conflict、double buffer、L2 cache 和核间负载不均衡。适用于算子已经正确但性能不足的场景。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC 性能分析与优化

在算子已经通过正确性验证后使用本 skill。不要在仍有精度错误、崩溃或 ABI 问题时做性能改写；先用 `ascendc-precision-debug` 或 `ascendc-crash-debug` 收敛到可验证版本。

## 1. Profiling 信号到优化动作

| 主要信号 | 常见瓶颈 | 优先动作 |
|---|---|---|
| Vector 占比高 | 向量计算受限 | 融合 UB 内计算阶段，减少 Cast，替换高开销向量序列 |
| MTE2 时间高 | GM 到 UB 搬运受限 | 增大单次搬运粒度，检查 32B 对齐，启用 double buffer |
| MTE3 时间高 | UB 到 GM 写回受限 | 减少中间写回，尽量一次写出最终结果 |
| Cube 占比高 | 矩阵计算受限 | 提升 L1/L0 复用，保留 epilogue 融合 |
| Scalar 占比高 | 索引和启动开销受限 | 将循环不变量移到 host tiling，拆出常见 shape 快路径 |
| 核间耗时差异大 | block/tail 分配不均 | 重新检查 `blockDim`、每核长度和 tail 公式 |
| UB bank conflict 高 | UB 访问冲突 | 调整 LocalTensor 偏移、stride 或 padding |

## 2. Vector Bound

优先检查：

- 多个 UB 计算阶段是否可以在一次 `CopyOut` 前完成。
- fp16/bf16 与 fp32 之间是否存在重复 Cast。
- 敏感数学链路是否只在必要位置升精度。
- `where`、比较、乘法 mask 是否引入额外临时张量。
- reduction 是否使用当前 SDK 中更合适的向量归约 API，而不是 scalar loop。

如果参考语义要求 fp32 中间结果，不要为了速度直接改成 fp16 计算。

## 3. MTE Bound

检查顺序：

1. 每次 `DataCopy` 的元素数是否足够大，可以摊薄 DMA setup 开销。
2. GM 地址和 UB 地址是否尽量满足 32B 对齐。
3. 非对齐路径是否只在 tail 或特殊 shape 中使用 `DataCopyPad`。
4. 队列深度是否导致 MTE2 和 Vector 串行执行。
5. 输入是否存在跨 tile 复用，可以少读一次 GM。

double buffer 的目标是让搬运和计算重叠。若 trace 仍显示串行，先检查队列配对和依赖，再调 tile size。

## 4. Scalar Bound

小 shape、broadcast、index/gather 类算子容易被 scalar 索引开销主导。

常用处理：

- 在 host tiling 中预计算 stride product、shape flag、公共偏移。
- 为 contiguous、scalar broadcast、last-dim broadcast 建立专门快路径。
- 对很小 workload 降低 `blockDim`，避免启动过多核。
- 避免在 kernel 内重复计算 `div/mod` 链；必要时改变任务划分。

## 5. Bank Conflict

当 profiling 指向 UB bank 或 bank-group conflict：

- 让高频 LocalTensor 的起始偏移至少错开 32B。
- 避免读写热点操作数落在同一段 UB 区域。
- 重新确认 vector repeat stride、block stride 的单位。
- padding 会增加 UB 占用，只在冲突收益大于容量损失时使用。

## 6. L2 与缓存策略

cache hint 只在存在数据复用时使用：

- 多次读取的只读输入可以启用普通缓存策略。
- 一次性 streaming 数据不应污染 L2。
- matmul-like 算子优先通过 L1/L0 复用减少 GM 访问。

## 7. 批量优化纪律

批跑时每轮只改一个性能假设：

```text
op:
baseline correctness:
profile symptom:
changed variable:
tested shapes:
speedup/regression:
next action:
```

不要把 tolerance 修改、dtype 覆盖收缩和 kernel 优化混在同一轮。若一个优化只提升单个 shape 但显著回退多数 shape，应回退或拆成 shape-specific 路径。

## 8. 样本反馈与窄路径

逐样本性能表可以用来定位退化路径，但不要把裸 shape 特化伪装成通用 kernel 优化。只有在某个 dtype、rank、layout、broadcast、对齐或数值分布模式明确主导总耗时，且通用路径重写风险较高时，才考虑窄路径。

- 优先把条件写成语义规则：dtype、rank、contiguous、same-shape、last-dim broadcast、对齐状态、特殊数值模式；裸 shape 常量只作为临时诊断或已知输入集合的兜底手段。
- 窄路径必须保持全 shape 覆盖，不能收缩 dtype/layout 支持，不能绕开正确性验证。
- 每次只引入一条窄路径，并同时比较全量指标和逐样本指标；如果总指标不优，即使单个样本变快也应回滚。
- 若窄路径稳定收益明显，再考虑把它上升为通用 tiling、kernel 分支或 host dispatch 规则，而不是继续堆叠更多裸 shape 判断。
- 库实现旁路只能作为退化路径替代或性能上界参照；它适合救 dtype 转换、复杂广播、极小规约或特殊值分布异常的场景，不等价于完成了 AscendC 通用优化。

推荐把窄路径条件写成“语义谓词”，而不是把完整输入 shape 写死：

```cpp
static bool IsLargeSameShapeInt8(const at::Tensor& a, const at::Tensor& b) {
  return a.scalar_type() == at::kChar &&
         b.scalar_type() == at::kChar &&
         a.dim() == b.dim() &&
         a.sizes() == b.sizes() &&
         a.is_contiguous() &&
         b.is_contiguous() &&
         a.numel() >= (1 << 24);
}

static bool IsLastDimBroadcastHalf(const at::Tensor& x,
                                   const at::Tensor& bias) {
  return x.scalar_type() == at::kHalf &&
         bias.scalar_type() == at::kHalf &&
         x.dim() >= 2 &&
         bias.dim() == 1 &&
         bias.size(0) == x.size(x.dim() - 1) &&
         x.is_contiguous() &&
         bias.is_contiguous();
}

at::Tensor op(const at::Tensor& x, const at::Tensor& y) {
  if (IsLargeSameShapeInt8(x, y)) {
    // 库实现旁路只覆盖明确退化的模式，其余输入仍走 AscendC 主路径。
    return at::maximum(x, y);
  }
  return launch_ascendc_kernel(x, y);
}
```

特殊数值模式也应写成语义条件。例如 all-zero、all-NaN、identity segment、single-segment 可以有快路径，但必须保证输出语义完整：

```cpp
if (is_all_zero && op_is_multiplicative_or_activation_zero_preserving) {
  return at::zeros_like(input);
}

if (is_same_size_resize && input.scalar_type() == output_dtype) {
  return input.contiguous();
}
```

### 8.1 从 per-shape 退化到通用分桶

per-shape 优化的正确打开方式不是“哪个 shape 慢就写哪个 shape”，而是先把慢样本归因到可复用的语义分桶。常见分桶包括：

| 退化样本特征 | 更通用的分桶条件 | 常见处理 |
|---|---|---|
| 单个超大 same-shape int8/uint8 elementwise 很慢 | integral dtype、same-shape、contiguous、numel 很大 | 避免 fp32 round-trip，增加 native integer path 或库旁路 |
| rank 较高的 half/bf16 逐元素算子慢 | non-float dtype、rank >= 4、contiguous、无广播 | 调大 tile，减少 cast buffer，必要时 dtype 专门路径 |
| `(N, D)` 与 `(D,)` 广播慢 | last-dim broadcast、bias contiguous、D 对齐或接近对齐 | host 侧标记 broadcast 模式，kernel 中按行复用小输入 |
| 小 D 规约慢 | reduction dim 很小、row 数很多 | 合批多行，或用标量规约绕开重同步 |
| 非 last-dim 规约慢 | reduce axis 不连续、stride 大 | 先转化布局、分轴专门路径，或使用库实现作为退化路径 |
| index/scatter/gather 特定 case 慢 | 索引形成连续段、identity reduce、single segment | 连续 DataCopy、分段合并、避免原子或读改写 |
| 特殊值样本极慢 | all-zero、all-NaN、constant、identity segment | 语义快路径直接填充、拷贝或跳过计算 |

推荐流程：

1. 先按 `gen_us / reference_us` 或绝对耗时找出主导样本。
2. 对慢样本记录 dtype、rank、contiguous、broadcast、reduce axis、numel、对齐、特殊值分布。
3. 找到能解释多个慢样本的共同条件，再写 host dispatch 或 tiling mode。
4. 如果只能解释一个样本，先把条件写成最窄语义谓词，并在注释中说明它代表的模式，而不是说明具体 shape。
5. 每次只启用一个新分桶，观察全量样本是否被拖慢。

host 侧可以把语义分桶编码成 tiling mode，避免 device 端反复解析 shape：

```cpp
enum class ElemMode : int32_t {
  kGeneric = 0,
  kSameShapeContiguous = 1,
  kLastDimBroadcast = 2,
  kLargeIntegralSameShape = 3,
};

static ElemMode ClassifyElementwise(const at::Tensor& x,
                                    const at::Tensor& y) {
  const bool sameShape = x.sizes() == y.sizes();
  const bool bothContig = x.is_contiguous() && y.is_contiguous();
  const bool integral = at::isIntegralType(
      at::promote_types(x.scalar_type(), y.scalar_type()),
      /*includeBool=*/false);

  if (sameShape && bothContig && integral && x.numel() >= (1 << 24)) {
    return ElemMode::kLargeIntegralSameShape;
  }
  if (sameShape && bothContig) {
    return ElemMode::kSameShapeContiguous;
  }
  if (x.dim() >= 2 && y.dim() == 1 &&
      y.size(0) == x.size(x.dim() - 1) && bothContig) {
    return ElemMode::kLastDimBroadcast;
  }
  return ElemMode::kGeneric;
}

tiling.mode = static_cast<int32_t>(ClassifyElementwise(x, y));
tiling.inner = x.size(x.dim() - 1);
tiling.total = x.numel();
```

device 端再用 mode 选择轻量分支：

```cpp
if (tiling->mode == static_cast<int32_t>(ElemMode::kSameShapeContiguous)) {
  ProcessContiguous();
} else if (tiling->mode == static_cast<int32_t>(ElemMode::kLastDimBroadcast)) {
  ProcessLastDimBroadcast();
} else {
  ProcessGeneric();
}
```

当必须暂时用完整 shape 常量时，应把它包在更外层的语义谓词后面，并且只作为兜底保护：

```cpp
static bool IsKnownPathologicalLarge5DHalf(const at::Tensor& x) {
  if (x.scalar_type() != at::kHalf || !x.is_contiguous() || x.dim() != 5) {
    return false;
  }
  // 只作为固定输入集合中的兜底；后续应替换成 rank/numel/tile 模式规则。
  return x.numel() > (1 << 22) && x.size(3) % 32 != 0;
}
```

## 9. 常见可复用优化范式

从多类 elementwise、reduction、index、normalization 和 geometry 算子的优化记录中，收益较稳定的范式如下：

- **去掉冗余转换和恒等计算**：删除 fp32 add-zero copy、重复 `ToF32/FromF32`、无用 `Muls/Adds`、死分支和不用的 TQue。若 dtype 已经满足计算要求，直接以输入 LocalTensor 作为算子源。
- **融合相邻向量阶段**：把 `Muls+Add` 改为 `Mad`，把 softplus/activation 的中间步骤合并，或把 epilogue 中的 scale、bias、cast 合并到最后一次写出前完成。
- **按 UB 预算调 tile**：tile 长度应按 dtype、scratch buffer 数、queue 深度和双缓冲需求计算；收益通常来自减少 tile 次数，但要防止 UB 溢出和队列容量互相挤占。
- **让搬运和计算重叠**：优先尝试 queue depth 2/3、double/triple buffer、提前 CopyIn 下一 tile、减少手写 `SetFlag/WaitFlag`。若只是同一阶段串行排队，增加 buffer 深度不会自动带来收益。
- **只在必要处同步**：规约或标量回读后能用 `PipeBarrier` 的地方，通常比成对事件旗更轻。删除同步前要确认后续读写确实没有跨流水依赖。
- **批量化小工作单元**：窄行 softmax、argmax、cross entropy、foreach 和 SwiGLU 小行场景，常见收益来自把多行/多 tensor 合到一个 UB tile 或一次 kernel call，摊薄启动、同步和 CopyOut 成本。
- **用连续块替代标量读写**：index/gather/scatter/resize 中，能把 `GetValue/SetValue` 改成批量 DataCopy、连续 CopyOut、成组 lane 更新或 UB 暂存后再写回时，收益通常远大于微调算术。
- **把索引算术移出内层**：将 `div/mod`、stride product、row base、w-table base、segment base、固定维度判断移到 host tiling、Init 或 batch 开头；内层尽量用增量 offset 和 int32 计数器。
- **利用特殊数值和结构模式**：all-zero、all-NaN、single-segment、identity segment、same-size resize、two-class cross entropy、small reduction 等模式可以有独立快路径，但条件必须来自语义而非偶然 shape。
- **减少 GM 往返和中间写回**：能在 UB 中复用输入、gamma、cos/sin、row cache、partial sums 时，优先缓存；最终结果尽量一次写出，避免中间张量先写 GM 再读回。
- **合理选择标量或向量规约**：很小 last-dim 上硬件向量规约可能被同步成本淹没，标量累加反而更快；大行、多行或可批量规约时再优先使用向量 Reduce。
- **清理死代码也要有性能假设**：删除未使用 include、成员、entry、helper、mode 分支有时能缩小编译产物并改善调度，但应作为低风险小步验证，不要替代真正的瓶颈优化。

### 9.1 去冗余 copy、cast 和恒等计算

常见坏味道是为了统一 dtype 路径，float 输入也先做一次 `Adds(x, 0)` 或 `Cast(float->float)`。这会多占一个 UB buffer 和一次 vector pass。

```cpp
// 较差：float32 路径也复制一遍。
auto xLocal = xQ.DeQue<T>();
auto xf = calcBuf.Get<float>();
if constexpr (std::is_same_v<T, float>) {
  Adds(xf, xLocal, 0.0f, count);
} else {
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
}

// 更好：float32 直接使用输入；非 float 才转换。
auto xLocal = xQ.DeQue<T>();
if constexpr (std::is_same_v<T, float>) {
  ComputeFloat(xLocal, count);
} else {
  auto xf = calcBuf.Get<float>();
  Cast(xf, xLocal, RoundMode::CAST_NONE, count);
  ComputeFloat(xf, count);
}
```

类似地，固定系数可以在 host 或 `Init` 中预折叠，避免每 tile 重复做标量乘加：

```cpp
// host/Init 阶段
tiling.effScale = scale * baseLog;
tiling.effShift = shift * baseLog;

// kernel 阶段
Muls(tmp, x, tiling.effScale, count);
Adds(tmp, tmp, tiling.effShift, count);
```

### 9.2 按 UB 预算选择 tile

tile 长度不要只看输入大小，还要把 dtype、scratch buffer、queue 深度和 double buffer 都算进去。经验上，先用安全预算，再按逐样本结果微调。

```cpp
static int64_t AlignDown(int64_t x, int64_t align) {
  return x / align * align;
}

int64_t PickTile(int64_t ubBytes, int64_t elemBytes,
                 int64_t inputBuffers, int64_t outputBuffers,
                 int64_t scratchBuffers, int64_t queueDepth) {
  int64_t buffers = (inputBuffers + outputBuffers) * queueDepth + scratchBuffers;
  int64_t usableBytes = ubBytes * 8 / 10;  // 留出队列和临时对象余量。
  int64_t tile = usableBytes / (buffers * elemBytes);
  return std::max<int64_t>(256, AlignDown(tile, 256));
}
```

如果一个算子同时有 fp32 和 fp16/bf16 路径，通常需要 per-dtype tile：

```cpp
tiling.tileLength =
    dtype == DTYPE_FLOAT ? PickTile(ubBytes, 4, 2, 1, 3, 2)
                         : PickTile(ubBytes, 2, 2, 1, 4, 2);
```

### 9.3 搬运和计算重叠

double/triple buffer 的关键不是“把 buffer 数加大”，而是循环顺序真的允许下一 tile 的 CopyIn 与当前 tile 的 Compute/CopyOut 重叠。

```cpp
constexpr int32_t BUFFER_NUM = 2;

for (int32_t i = 0; i < tileNum + BUFFER_NUM; ++i) {
  if (i < tileNum) {
    CopyIn(i % BUFFER_NUM, i);
  }
  if (i >= 1 && i - 1 < tileNum) {
    Compute((i - 1) % BUFFER_NUM, i - 1);
  }
  if (i >= 2) {
    CopyOut((i - 2) % BUFFER_NUM, i - 2);
  }
}
```

若 Compute 里马上等待 CopyIn 完成、CopyOut 又马上等待 Compute 完成，trace 仍会显示串行。此时应先检查队列 EnQue/DeQue 顺序和事件依赖，再调整 tile。

### 9.4 同步降噪

规约后只需要保证向量结果对标量回读可见时，`PipeBarrier` 往往比成对 `SetFlag/WaitFlag` 更轻：

```cpp
ReduceSum(sumLocal, xLocal, tmpBuf, count);
PipeBarrier<PIPE_V>();
float sum = sumLocal.GetValue(0);
```

不要机械删除所有同步。若后续跨 MTE/VEC/Scalar 流水读写同一 buffer，仍需保留正确的队列同步或事件同步。

### 9.5 批量化小行和多输出

小行 softmax、argmax、cross entropy、foreach 类算子常被 kernel launch、barrier 和 CopyOut 开销支配。可以把多行结果攒在 UB 中，再一次写回。

```cpp
constexpr int32_t ROW_BATCH = 16;
auto outLocal = outBuf.Get<float>();

for (int32_t rb = 0; rb < rows; rb += ROW_BATCH) {
  int32_t activeRows = Min(ROW_BATCH, rows - rb);
  for (int32_t r = 0; r < activeRows; ++r) {
    outLocal.SetValue(r, ComputeOneRow(rb + r));
  }
  DataCopy(outGm[rb], outLocal, activeRows);
}
```

foreach 类 Tensor[] 算子若大多数输入都走同一简单模式，优先考虑把这些输入合到一个 kernel 调用中；只有不规则或特殊值 tensor 单独分流。

### 9.6 用连续块替代标量读写

索引类算子容易退化成每元素 `GetValue/SetValue`。当索引映射在某个轴上形成连续段时，应先把连续段搬到 UB，再批量写回。

```cpp
// 较差：每个元素单独读写。
for (int32_t i = 0; i < count; ++i) {
  T v = xGm.GetValue(base + i);
  yGm.SetValue(outBase + i, v);
}

// 更好：连续段直接搬运。
DataCopy(local, xGm[base], count);
DataCopy(yGm[outBase], local, count);
```

对 gather/scatter，先识别 rank2、dim0、last-dim contiguous、identity reduce 等模式，再为这些模式写块化路径；通用路径保留完整语义。

### 9.7 索引算术外提

几何、resize、grid、scatter 类算子的内层 `div/mod` 和多级 stride 乘加很贵。把不变项外提到 batch 开头或 host tiling：

```cpp
// 较差：内层每个输出点都重复算。
int64_t n = linear / (OH * OW);
int64_t rem = linear % (OH * OW);
int64_t oh = rem / OW;
int64_t ow = rem % OW;
int64_t inBase = ((n * C + c) * IH + h0) * IW;

// 更好：按行推进，内层只增量更新。
int64_t rowBase0 = ((n * C + c) * IH + h0) * IW;
int64_t rowBase1 = ((n * C + c) * IH + h1) * IW;
for (int32_t ow = owStart; ow < owEnd; ++ow) {
  int64_t wBase = wTableBase + ow;
  ComputePixel(rowBase0, rowBase1, wBase);
}
```

计数器和偏移量如果能证明范围足够，优先用 `uint32_t` / `int32_t`，减少 64 位整数算术压力。

### 9.8 复用 UB 数据并减少 GM 往返

normalization、rotary、reduction 类算子常需要同一行数据或系数多次参与计算。若 UB 容量允许，优先缓存到专用 buffer：

```cpp
// pass 1: 读 x，累计 sumsq，同时缓存 x。
DataCopy(xLocal, xGm[rowBase], D);
DataCopy(xCache, xLocal, D);
ReduceSum(sumsq, Square(xLocal), tmp, D);

// pass 2: 直接复用 xCache，不再从 GM 读 x。
float rscale = Rsqrt(sumsq.GetValue(0) / D + eps);
Muls(xLocal, xCache, rscale, D);
Mul(outLocal, xLocal, gammaLocal, D);
DataCopy(yGm[rowBase], outLocal, D);
```

如果 D 很小，缓存整行可能不划算；如果 D 很大，分 tile 缓存可能挤掉必要 scratch。先按 UB 预算决定是否启用。

### 9.9 小规约的标量路径

硬件向量规约适合大行或多行批量规约；很小 last-dim 可能被同步和临时 buffer 成本淹没。

```cpp
if (D <= 32) {
  float sum = 0.0f;
  for (int32_t i = 0; i < D; ++i) {
    float v = static_cast<float>(xLocal.GetValue(i));
    sum += v * v;
  }
  outLocal.SetValue(0, PostProcess(sum));
} else {
  Mul(tmp, xLocal, xLocal, D);
  ReduceSum(sumLocal, tmp, reduceTmp, D);
  PipeBarrier<PIPE_V>();
  outLocal.SetValue(0, PostProcess(sumLocal.GetValue(0)));
}
```

### 9.10 数值敏感规约与量化边界

reduction 后接 `sqrt/div/round/cast` 的链路中，tile 长度和 partial sum
树形不只是性能参数，也会改变最后一两 ulp，进而把 int8/uint8 量化结果推过
半整数边界。调这类路径时要一次只改一个变量，并记录每个候选 tile 的
hard-mismatch 数；不要因为总体 MERE/MARE 很小就忽略整数输出的 exact gate。

```cpp
// 规约 tile 不只是吞吐参数；它会改变 partial sum tree。
constexpr uint32_t TILE = 256;
for (uint64_t off = 0; off < dim; off += TILE) {
  uint32_t count = Min<uint64_t>(TILE, dim - off);
  Mul(tmp, xLocal, xLocal, count);
  ReduceSum(partial, tmp, reduceTmp, count);
  partialBuf.SetValue(chunk++, partial.GetValue(0));
}
ReduceSum(total, partialBuf, reduceTmp, chunk);
```

如果一个量化算子在少量元素上只差 `+/-1`，优先检查：

- sum/amax 的规约树形是否和参考路径不同；
- scale 是 scalar 算出来的，还是通过 vector `Div/Muls` 后回读的；
- round mode 是否和参考一致，特别是 half-boundary；
- clamp 和 cast 的顺序是否改变了 NaN/Inf 或边界值。

attention、matmul epilogue 或卷积反传里若失败点全部集中在 small-value /
cancellation 区间，`abs(out) < eps -> 0` 只能作为诊断手段，不能直接当成
通用修复。它经常能降低最坏相对误差（MARE），但会把更多非零小值压成零，
导致整体 MERE 或 small-value mismatch 变差。只有当阈值来自明确语义
（例如全 mask 行、padding 贡献、已知零输入段），并且同时降低 MERE/MARE
与 mismatch 数时，才应保留。

特殊值分支要避免在 AICore scalar 路径里临时用 `inf/inf`、`0/0` 等算术造
NaN；这类写法可能触发运行异常，或在不同编译/硬件组合上不稳定。若语义确实
要求 NaN/Inf，优先使用明确的 bit-pattern 写入、输入 raw 值传播，或把特殊值
转换封装成单独 helper，并用只覆盖特殊值的 case 先验证。

手写卷积、池化或 stencil 类 scalar kernel 时，padding 越界分支不要机械地
`continue` 后认为等价于乘以 0。对普通有限输入这是对的，但当权重、bias 或
被 padding 位置对应的另一侧操作数含 NaN/Inf 时，库实现可能保留
`0 * inf -> NaN` 这类特殊值传播。遇到 `mere=0/mare=0` 但 NaN mask 不一致，
优先检查 padding 分支是否跳过了非有限操作数；修复时用语义条件传播明确的
quiet-NaN，不要用非法算术临时制造 NaN。

### 9.11 死代码清理的边界

删除死代码可以减少模板实例化、寄存器压力或二进制体积，但它不是主要优化手段。适合清理的对象包括：

- 已不可达的 mode 分支和 entry。
- 只在 `Init` 中消费、之后不再使用的成员变量。
- 不再使用的 TQue/TBuf、include 和 helper。
- 与当前 dtype 路径无关的 `if constexpr` 死分支。

清理后仍要检查 CMake、注册入口、host wrapper 和 kernel entry 是否一致，避免把 ABI 或覆盖范围一起删掉。
