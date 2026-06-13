# MathIR 到 Triton Ascend Lowering 语义指南

本文档用于 Coder 根据 MathIR 的 `formula`、`axis_mapping`、`symbol_binding` 和 `boundary_treatment` 识别语义形态。MathIR 是语义来源，不是固定 schedule；最终是否使用 `tl.dot`、output tile、grid-stride 或多 kernel，应结合 Ascend 后端约束、shape 和 verifier 反馈选择。

具体的 Ascend 调度优先级、UB/Grid 限制、ConvTranspose 大 shape 实现策略，见 `suggestion_docs.md`。

## 1. 先识别表达式类型

- 纯逐元素表达式：将输出元素映射到一维或二维 tile，使用 `tl.load` / `tl.store` 和向量化 elementwise 计算。不要退化成一个 program 只算一个标量。
- 普通 reduction：`sum/max/mean/var` 等单 tensor 归约优先用 `tl.sum`、`tl.max` 或分块两阶段 reduction；没有乘法 contraction 时不要强行套 `tl.dot`。
- GEMM-like contraction：公式中存在 `sum_k lhs(...) * rhs(...)`，且输出 parallel 轴能拆成位置轴 M 和通道/feature 轴 N 时，可以把它识别为 `tl.dot` 候选。A tile 语义形状为 `(BLOCK_M, BLOCK_K)`，B tile 语义形状为 `(BLOCK_K, BLOCK_N)`，accumulator 语义形状为 `(BLOCK_M, BLOCK_N)`。
- Conv-like contraction：`sum_{ci,kh,kw} input[...] * weight[...]` 可以视为 implicit-GEMM 候选，但 MathIR 只说明它是卷积乘加语义，不要求一定用 `tl.dot`。复杂 padding/stride/dilation、group/depthwise 或 ConvTranspose 可选择更稳的 schedule。
- 前缀扫描/递推表达式：`cumsum/cumprod` 或顺序依赖不是普通 reduction，也不能用 `tl.dot`；优先使用块内 scan 原语或分块两阶段 scan。

## 2. 从 MathIR 轴映射到 grid / tile

- `axis_mapping.parallel` 中的每个轴都是输出坐标的一部分。先决定哪些轴映射到 `tl.program_id`，哪些轴用 `tl.arange` 放进同一个 program 内并行计算。
- 每个输出坐标必须只由一个 program 写入。若多个 parallel 轴展平成线性轴，grid 解码、地址计算和 store mask 必须使用同一套快变维规则。
- `axis_mapping.reduction` 不直接决定输出 grid。它应在 program 内被向量化归约、作为 `tl.dot` 的 K tile、或拆成两阶段 partial reduction。
- 当 parallel 轴自然分成“输出位置”和“输出通道/feature”两组时，可以把位置组展平成 M，把通道/feature 组展平成 N，把 reduction 轴展平成 K。这只是 `tl.dot` 的 tile 约定，不能反过来强迫所有表达式都变成 M/N/K。
- 对 depthwise/grouped conv，parallel 通道轴通常同时决定输入通道和输出通道归属。只有 group 内能形成 dense `(K, N)` 权重 tile 时才使用 `tl.dot`。

## 3. 边界与 mask 语义

- `boundary_treatment` 决定哪些 lane 是有效贡献。padding、输入反推越界、ConvTranspose 的 stride 整除条件、group/channel 合法性、tail M/N/K 都应组合进 load/store mask。
- “zero padding / 超出输入范围视为 0”：把输入反解坐标合法性合进 load mask，`other=0.0`。
- max/min reduction 的无效 lane 要使用匹配的 neutral value：max 用 `-float("inf")`，min 用 `float("inf")`，sum/dot 用 `0.0`。
- softmax/attention 即使语义上没有 causal/padding mask，也要用 `offs < extent` 屏蔽 tile tail，并把无效 lane 置为 `-inf`。
- mask 必须在 load 前阻止越界地址参与内存访问，不能先越界 load 再用 `tl.where` 修正数值。

## 4. `tl.dot` 语义判断

可以考虑 `tl.dot` 的语义条件：
- 公式包含乘法加和 contraction：`sum_k lhs(...) * rhs(...)`。
- 一个 program 能覆盖多个真实 M 位置和多个 N 通道/feature。
- K 可以展平成一维 `K_total`；尾部用 padded `BLOCK_K` 和 mask 过滤。
- 输出 tile 能复用输入或权重 tile，例如多个空间位置共享同一组 weight，或多个输出通道共享同一组 input patch。

不应从 MathIR 强行推出 `tl.dot` 的场景：
- reduction 是 max/min/mean/variance，但没有两个输入的乘法 contraction。
- depthwise/grouped conv 中每个输出通道只依赖极少输入通道，无法形成有效 dense `(BLOCK_K, BLOCK_N)` tile。
- K 很小且 N 也很小，可以用多输出 tile + 小 outer-product；但不能退化成一输出一 program 的标量循环。
- ConvTranspose 的 MathIR 公式描述的是语义，不能强迫生成 output-to-input `tl.dot`。若后端调度风险高，应按 `suggestion_docs.md` 选择 output-centric tile、input-centric scatter-accum 或 grid-stride。

## 5. 常见 lowering 模式

### Elementwise / Activation

- 线性 offsets 映射到输出元素，`mask = offsets < total`。
- 对二维输入可把 `M, N` 展平成一维，也可用 `pid_m/pid_n` 做 2D tile。
- 简单函数如 Softsign 应直接按公式 `x / (1 + abs(x))` 计算；如果当前 API 文档没有某个高层数学函数，就用等价基础算子展开。

### Matmul / Linear

- 对 `Y[m,n] = sum_k A[m,k] * B[k,n]`，A tile 为 `(BLOCK_M, BLOCK_K)`，B tile 为 `(BLOCK_K, BLOCK_N)`，accumulator 为 `(BLOCK_M, BLOCK_N)`。
- PyTorch `nn.Linear` 的 weight 布局是 `(O, I)`。计算 `X(N,I) @ W^T` 时，B tile 按 `W[o, k]` 读取，逻辑形状为 `(BLOCK_K, BLOCK_O)`。
- 如果需要预处理 weight，应在 `ModelNew.__init__` 中一次性完成；`forward()` 不要反复 `permute/contiguous`。

### Conv2d implicit-GEMM

- M = `N_batch * H_out * W_out`，N = `C_out`，K = `C_in * K_h * K_w`。
- A tile 通过 `(offs_m[:, None], offs_k[None, :])` 反解 `n/oh/ow/ci/kh/kw` 后 masked load input，形状 `(BLOCK_M, BLOCK_K)`。
- B tile 通过 `(offs_k[:, None], offs_n[None, :])` 从原始 `W[co,ci,kh,kw]` 读取，形状 `(BLOCK_K, BLOCK_N)`。
- padding、stride、dilation、tail K 都放进 mask。bias、activation、scale、clamp 等 cheap epilogue 可在 accumulator 上融合。

### ConvTranspose2d / ConvTranspose3d

- MathIR 公式是语义来源，不要求按公式字面生成 output-to-input `tl.dot`。
- 2D 权重布局是 `(C_in, C_out, K_h, K_w)`；3D 权重布局是 `(C_in, C_out, K_d, K_h, K_w)`。
- output-centric 实现需要从输出坐标反推输入坐标：通用情况先算 `num_h = oh + padding_h - kh * dilation_h`，满足 `num_h % stride_h == 0` 后 `ih = num_h // stride_h`，W/D 方向同理。
- input load mask 同时包含 stride 整除、输入 bounds、channel/group 合法性和 tile tail。
- store 地址必须包含所有 block 起点，例如 `co_start/w_start/d_start/h_start`；只使用相对 offsets 会覆盖前面的 block。

### Pool / Norm / Activation 融合

- Conv/GEMM 是 heavy producer，bias、scale、activation、hard clamp、小窗口 pool、small fixed reduction 是 cheap consumer，能在寄存器中融合时优先融合。
- GroupNorm / LayerNorm / RMSNorm 的核心是组内统计 reduction，通常不是 `tl.dot` contraction；如果前面有 producer tile，可以在 producer tile 上做 `sum/sum_sq/max`，否则使用专门 reduction kernel。
- 如果融合导致 mask、布局或寄存器压力过高，可以分阶段：第一个 Triton kernel 写出 heavy producer，第二个 Triton kernel 处理 cheap post-op。

## 6. MathIR lowering 检查

- A tile `(BLOCK_M, BLOCK_K)` 中，位置字段用 `[:, None]`，K 字段用 `[None, :]`。
- B tile `(BLOCK_K, BLOCK_N)` 中，K 字段用 `[:, None]`，N/channel 字段用 `[None, :]`。
- store tile `(BLOCK_M, BLOCK_N)` 中，位置字段用 `[:, None]`，通道字段用 `[None, :]`，store mask 写成 `mask_m[:, None] & mask_n[None, :]`。
- `acc += A * B` 不能替代 `tl.dot(A, B)`，因为它会混淆 K 轴和输出通道轴。
- 反过来，`tl.dot` 也不能替代所有乘加 contraction。若 `tl.dot` 造成 UB/root-alloc/编译器崩溃，应换 schedule，而不是继续放大 tile 或强行改 mask。
