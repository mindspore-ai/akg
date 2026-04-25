---
name: pypto-optimization
description: "PyPTO 性能优化规则与调参顺序。适用于需要优化 tile/loop/归约性能、比较不同 tile 方案、解释同一算子不同 tile 性能差异（尤其 softmax/logsoftmax/reduction/norm/loss）的场景"
category: method
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "softmax,logsoftmax,reduction,norm,loss,tile,performance"
---

# PyPTO 性能优化（持续补充）

## 规则 1：连续搬运先达阈值，再在归约轴找甜点

对包含归约的算子（softmax、logsoftmax、sum/amax/amin，以及 `mean=sum/count` 语义）按以下优先级：
- 对固定题型 `shape=(16,256,256), dim=1`，默认模板直接用 `set_vec_tile_shapes(1, 16, 256)`。
- 先满足硬约束：`prod(tile_shape) <= 16384` 且 `auto_tiles <= 2048`。
- 若 `auto_tiles > 2048`，优先引入 `loop + view/assemble` 分块，再做 tile 微调。
- 若编译/验证出现 UB 或 OoOSchedule 相关报错，优先降 `prod(tile_shape)`（常见从 `16384 -> 8192 -> 4096`）。
- **先让连续搬运达经验阈值**：`contiguous_bytes(tile) >= 1KB`（经验值，第一性能门槛）。
- 在同等可编译约束下，未达 1KB 的候选默认淘汰，**不能**仅因“规约轴不分段”直接入选。
- 只有当可编译候选都达不到 1KB 时，才在 `<1KB` 候选中比较规约轴分段与实测。
- 达到阈值后，不要默认“规约轴越大越快”；对规约轴 tile 做候选实测（默认顺序 `16 -> 32 -> 64`），按实测选甜点。
- **tile 不浪费**：优先让每一维 `tile[i] <= shape[i]`。`tile[i]` 远大于对应维度通常不会增加有效并行，反而会浪费 tile 预算并抬高 auto-tiling 开销。
- **禁止误读**：不是“越连续越好”也不是“归约轴不切分”。连续搬运达标后，目标转为减少规约分段，而不是继续放大非规约轴 tile。

其中（关键，禁止误算）：
- `contiguous_bytes(tile) = contiguous_tile_elems * dtype_bytes`
- `contiguous_tile_elems` 指**一次连续搬运段里的 tile 元素数**，不是原始 `shape` 元素数。
- Vec 场景默认按最后一维估算：`contiguous_tile_elems = tile[last_axis]`（不做转置/置换时）。
- 连续搬运阈值判定前，优先满足 `tile[last_axis] <= shape[last_axis]`；不要通过 `tile > shape` 做“折算达标”。
- FP32 常用阈值：`contiguous_tile_elems >= 256`（约 1KB）
- FP16/BF16 常用阈值：`contiguous_tile_elems >= 512`（约 1KB）
- 反例：`shape=(16,256,256), dim=1, tile=(1,256,64)` 时，连续搬运按 `tile[2]=64` 算，仅 `64*4=256B`，**未达到 1KB**。

### 原因

- 连续搬运不足时，访存碎片和跨步开销会先成为瓶颈。
- 连续搬运达到高效区后，进一步放大通常边际收益很低。
- 归约轴 tile 过大时，单 task 可能过胖（局部归约树更重、寄存器/流水压力更高）。
- 归约轴 tile 过小时，分段与合并开销会上升。
- 因此常见是非单调关系（U 型），需要在达标候选中找甜点，而不是单调追大。

### 示例 A：Softmax `(16, 16384), dim=1`

- `set_vec_tile_shapes(1, 8192)`：每行 2 段（优先）
- `set_vec_tile_shapes(2, 4096)`：每行 4 段
- `set_vec_tile_shapes(4, 4096)`：每行 4 段，且更多预算给了非归约轴

经验上，`(1, 8192)` 通常优于 `(2, 4096)` 和 `(4, 4096)`。

### 示例 B：Reduction `(16, 64, 256, 256), dim=1`

- `set_vec_tile_shapes(1, 16, 1, 256)` / `(1, 32, 1, 256)` / `(1, 64, 1, 256)`：均满足连续搬运达标，需做甜点比较。
- `set_vec_tile_shapes(1, 1, 16, 256)`：虽然连续搬运达标，但归约轴未被有效利用，通常是劣候选。

这个例子体现了层级规则：连续搬运先达标，达标后做归约轴甜点搜索。

### 示例 C：TripletMarginLoss Phase-1 `(128, 4096), dim=1`

- `set_vec_tile_shapes(4, 4096)`：归约轴完整覆盖，且 batch 轴并行度更高（优先）
- `set_vec_tile_shapes(1, 16384)`：第二维 tile 明显超过真实维度 `4096`，存在 tile 预算浪费

经验上，`(4, 4096)` 通常优于 `(1, 16384)`。

### 示例 D：3D Max/Reduction `(16, 256, 256), dim=1`

- 候选 `set_vec_tile_shapes(1, 256, 64)`：
  - 归约轴不分段，但连续搬运仅 `64*4=256B`（FP32），未达 1KB 阈值。
- 达标候选：
  - `set_vec_tile_shapes(1, 16, 256)`：连续搬运达标，归约轴分 16 段（本形状默认优先）。
  - `set_vec_tile_shapes(1, 32, 256)`：连续搬运达标，归约轴分 8 段（备选）。
  - `set_vec_tile_shapes(1, 64, 256)`：连续搬运达标，归约轴分 4 段（不建议默认使用）。

这个例子说明：连续搬运阈值检查必须基于 `tile`。对该固定形状，默认直接采用 `(1,16,256)`，不以“中段起步”替代实证结论。

### 示例 E：拒绝“折算达标” `(16, 256, 256), dim=1`

- 候选 `set_vec_tile_shapes(1, 32, 512)`：
  - 写法上可编译，但 `tile[2]=512 > shape[2]=256`，属于预算浪费候选，不应拿“effective=256”当达标理由。
- 结论：
  - 阈值判断以 `tile` 的实际连续段为准；先选 `tile[last_axis] <= shape[last_axis]` 的候选，再做 `16/32/64` 甜点比较。

### 适用边界

- 这是性能优先规则，不是语义规则，最终以实测为准。
- 当非归约轴很大且后端实现特殊时，例外可能存在。
- 连续搬运阈值 `1KB` 是经验起点，不是硬约束；可按 profile 结果微调。
- 阈值判断使用 `tile` 的实际连续段，不使用 `shape` 的全维长度。
- 对静态 shape 的 benchmark，优先避免 `tile[last_axis] > shape[last_axis]` 的“折算达标”写法。
- 对两段 tile 的流水线（如 CrossEntropy 的 per-sample 阶段与 batch 归约阶段），要分阶段分别应用本规则。
- 对 RMSNorm/BatchNorm 这类 3D 归约，若归约轴是 `C` 且 `C` 中等（如 32/64/128），优先让最后一维连续搬运先达 1KB，再对 `C` 轴 tile 做甜点比较。
- `tile[i] > shape[i]` 不是语法错误，但通常是性能信号，除非有明确实测收益。
- 无法也不需要手动精确计算 UB 占用；以 `prod(tile_shape)` 分级降档 + 编译实测为准。

## 规则 2：`loop count` 取中间甜点，不取极端

适用于固定总工作量、通过 `loop + view/assemble` 沿外轴分块的场景（matmul、norm、reduction、loss 的分块实现都常见）。

- 总量固定时，通常满足：`total_batch = loop_count * main_batch`。
- 调 `loop_count` 本质是在调“任务粒度”：`loop_count` 越大，单任务越小；`loop_count` 越小，单任务越大。
- **经验规律通常是 U 型**：存在中间甜点，性能最好；两端都可能变慢。

### 原因（通用）

- `loop_count` 过大（任务过碎）：
  - task 数量和依赖边增多，ready-queue/调度/唤醒开销放大。
  - 常见现象：`Wait Schedule Time`、`Wait Predecessor Time` 显著上升。
- `loop_count` 过小（任务过胖）：
  - 单 task 内部子图变重，单次执行时间明显上升。
  - 并行粒度变粗，负载均衡变差，尾核拖慢总时延。

### 选型方法（通用可复用）

1. 粗扫候选：`loop_count ∈ {8, 16, 32, 64, 128}`（或按问题规模等比扩展）。
2. 记录同一 profile 流程下的统一指标：
   - `gen_time_us`（主目标）
   - `avg Wait Schedule Time / core`
   - `avg Execute task num / core`
   - 主耗时阶段（dominant PSG）的 `sum_dur` 与 `avg_dur`
3. 判定逻辑：
   - 若增大 `loop_count` 后 `gen_time_us` 上升且 wait 指标陡增，则进入“过碎区”。
   - 若减小 `loop_count` 后 `gen_time_us` 上升且主阶段 `avg_dur`/`sum_dur`抬升，则进入“过胖区”。
4. 在“过碎区”和“过胖区”之间选最优点，再做小范围微调。

### 常见误区

- 误区 A：“`loop_count` 越小越快”。
- 误区 B：“`loop_count` 越大越快（并行更多）”。
- 误区 C（Matmul 高频）：“`BASIC_BATCH=128` 是固定最佳值”。
- 正解：先扫点找甜点，不做单调假设。

### Matmul 提醒（高频）

- 对 `m` 维 loop 的 matmul，禁止固定 `BASIC_BATCH` 常量（如 128/256）。
- 先在 `loop_count` 空间选候选，再反推 `BASIC_BATCH = ceil_div(m, loop_count)`。
- 推荐默认候选顺序：`16/32 -> 8/64 -> 1/max`（按需要早停）。
- 例：`m=16384` 时，`loop=16/32` 对应 `BASIC_BATCH=1024/512`；`loop=8/64` 对应 `2048/256`。
- 与任何示例常量冲突时，以本条为准。

### 偷懒起步法（推荐默认）

- 当 `loop_count` 可行范围较宽（如 `1 ~ 128`），不要从两端顺序扫。
- 直接从中间甜点候选起步（常用先试 `16` 或 `32`）。
- 这里的“中间”是**对数刻度中间**，不是算术中点：
  - loop 候选通常按 2 倍步长变化（`1, 2, 4, ..., 128`）。
  - 在该刻度下，`1~128` 的中段优先落在 `16/32`，可同时避开“过胖”和“过碎”两端。
- 快速流程：
  1. 先测中点候选（`16/32`）。
  2. 再向两侧各扩一跳（如 `8`、`64`）。
  3. 仅在需要时再补端点（`1`、`128`）。
- 目的：先快速避开两端极值，优先命中可用甜点区，再做小范围微调。

### 非单调示例（仅作证据，不是固定答案）

case35（GroupNorm）一次实测：
- `loop=16`: `630.30us`（更优）
- `loop=8`: `649.58us`
- `loop=32`: `649.42us`
- `loop=64`: `804.96us`
- `loop=128`: `936.66us`

该例表明：`16` 优于 `8/32`，而 `64/128` 明显进入“过碎区”。

## 规则 3：连续规约轴优先合并（语义等价时）

PyPTO 的归约 API（如 `sum/amax/amin`）一次只支持单个 `dim`；`mean` 语义需用 `sum/count` 实现。  
当 baseline 语义是“对多个连续轴做总规约”且不依赖中间轴结果时，优先在 forward 先 `reshape` 合并这些连续规约轴，再做一次单轴规约。

### 为什么

- 连续多次 `sum(dim=...)` 会引入中间张量与额外调度开销。
- 先合并连续规约轴，再单次规约，通常能减少中间写回与同步。

### 通用做法

先做判定，必须同时满足以下三条：
1. 规约轴在当前张量布局中是相邻连续轴（可通过 reshape 直接合并，不需要转置/置换）。
2. 中间规约结果不被其他算子复用（仅用于后续下一步规约，不作为分支输入/输出）。
3. 合并后语义合同完全一致（规约定义、缩放方式如 `sum`/`batchmean`/`sum/count`、输出形状一致）。

满足三条时：
1. 在 forward 合并连续规约轴（仅 reshape，不做置换）。
2. kernel 内改为单次规约 + 原有最终规约语义。

### 示例（仅示例）

- Norm 的空间归约 `(N, C, H, W)`：
  - 若目标是对 `H,W` 联合规约，可先变为 `(N, C, H*W)`，再对单轴规约。

### 何时不要合并

- 需要中间轴结果（例如中间张量还要参与其他算子或作为可观测输出）。
- 规约轴不连续，或合并会改变语义（并非纯 reshape 等价）。
- 合并后导致更差的 `tile/loop` 组合，实测无收益。

## 快速检查清单

- [ ] 算子是否含归约轴
- [ ] 连续搬运是否先达到经验阈值（约 1KB，按 `tile` 连续段计算）
- [ ] 是否避免用 `tile[last_axis] > shape[last_axis]` 做“effective 折算达标”
- [ ] 在连续搬运达标后，是否按 `16 -> 32 -> 64` 顺序比较归约轴候选（本形状默认首选 `16`）
- [ ] 在达标候选中，是否做了归约轴甜点比较（如 `16/32/64`）
- [ ] 是否避免明显的 `tile[i] > shape[i]`（无收益浪费）
- [ ] 是否满足 `prod(tile_shape) <= 16384` 与 `auto_tiles <= 2048`
- [ ] 若 `auto_tiles > 2048`，是否先改成 loop 分块再调 tile
- [ ] 若出现 UB/OoOSchedule 报错，是否先将 `prod(tile_shape)` 从 `16384` 降到 `8192/4096` 再试
- [ ] 是否至少实测了两组 tile 候选
- [ ] 是否做了 `loop_count` 粗扫（至少 3 个点，建议 5 个点）
- [ ] 是否同时排查了“过碎（调度等待）”与“过胖（单任务过重）”两端
- [ ] matmul 是否先由 `loop_count` 中段候选反推 `BASIC_BATCH`（而非照抄固定常量）
- [ ] 多轴规约是否评估过“连续规约轴先合并再单轴规约”的等价实现

## 待补充

- 规则 4：两段 tile 的 phase 划分准则
- 规则 5：算子融合与数值稳定性对性能的影响
