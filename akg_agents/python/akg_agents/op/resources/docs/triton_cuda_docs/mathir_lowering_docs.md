# MathIR 到 Triton CUDA Lowering 技巧

本文档用于 Coder 根据 MathIR 的 `formula`、`axis_mapping`、`symbol_binding` 和 `boundary_treatment` 选择 Triton CUDA 实现方式。MathIR 是语义来源，不是固定 schedule；如果某个 lowering hint 与当前 shape、硬件或 verifier 反馈冲突，应保持语义等价并选择更合适的 Triton lowering。

## 1. 先识别表达式类型

- 纯逐元素表达式：将输出元素映射到一维或二维 tile，使用 `tl.load` / `tl.store` 和向量化 elementwise 计算。
- 普通 reduction：如果公式是 `sum/max/mean` 等单 tensor 归约，使用 `tl.sum`、`tl.max` 或手写分块归约；不要强行套 `tl.dot`。
- GEMM-like contraction：如果公式中存在 `sum_k lhs(...) * rhs(...)`，并且输出 parallel 轴能拆成位置轴 M 和通道/feature 轴 N，优先 lower 成 `tl.dot`。做法是把共享 reduction 轴展平为 K，让左输入按 `(offs_m, offs_k)` 组成 A tile `(BLOCK_M, BLOCK_K)`，右输入按 `(offs_k, offs_n)` 组成 B tile `(BLOCK_K, BLOCK_N)`，然后用 `acc += tl.dot(a, b)` 计算输出 tile `(BLOCK_M, BLOCK_N)`。Matmul、Linear、batched matmul，以及 `QK^T` / `PV` 这类双线性子表达式都属于这个类型；softmax、scale、mask 等应作为后续 cheap op 融合或分阶段处理。
- Conv-like contraction：如果公式是 `sum_{ci,kh,kw} input[...] * weight[...]`，本质上也是 GEMM-like contraction，但 A 矩阵来自隐式 im2col，不要真的 materialize。把输出位置 `(n, oh, ow)` 或 `(n, od, oh, ow)` 展平成 M，把输出通道 `co` 展平成 N，把 `(ci, kh, kw)` 或 `(ci, kd, kh, kw)` 展平成 K；A tile 用 `(offs_m[:, None], offs_k[None, :])` 反解 input patch 并 masked load 成 `(BLOCK_M, BLOCK_K)`，B tile 用 `(offs_k[:, None], offs_n[None, :])` 从 weight 读取成 `(BLOCK_K, BLOCK_N)`，再 `tl.dot(a, b)`。stride、padding、dilation 和 tail K 都放进 mask。
- 前缀扫描/递推表达式：如果公式是 `cumsum/cumprod` 或带顺序依赖的 prefix scan，不是普通 reduction，也不能用 `tl.dot`。优先使用块内 scan 原语或分块两阶段 scan。

## 2. 从 MathIR 轴映射到 grid / tile

- `axis_mapping.parallel` 中的每个轴都是输出坐标的一部分。先为这些轴选择 tile 覆盖方式：哪些轴映射到 `tl.program_id`，哪些轴用 `tl.arange` 放进同一个 program 内并行计算。
- 每个输出坐标必须只由一个 program 写入。若把多个 parallel 轴展平成一个线性轴，grid 解码、地址计算和 store mask 必须使用同一套快变维规则。
- `axis_mapping.reduction` 中的轴不直接决定输出 grid。它们应在 program 内被向量化归约、作为 `tl.dot` 的 K tile、或拆成两阶段 partial reduction；选择哪一种取决于 reduction 大小和是否存在乘法 contraction。
- 当 parallel 轴自然分成“输出位置”和“输出通道/feature”两组时，可以把位置组展平成 M，把通道/feature 组展平成 N；reduction 轴展平成 K。这只是 `tl.dot` 的 tile 约定，不能反过来强迫所有表达式都变成 M/N/K。
- 对 depthwise/grouped conv，parallel 通道轴通常同时决定输入通道和输出通道归属。不要把它直接当成 dense N 轴；只有 group 内能形成 dense `(K, N)` 权重 tile 时才用 `tl.dot`。

例子：`parallel=[b, j]`、`reduction=[i]`，公式为 `Y[b,j] = min_i X[b,i,j]`。
```python
pid_b = tl.program_id(0)
pid_j = tl.program_id(1)
offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

best = tl.full((BLOCK_J,), float("inf"), tl.float32)
for i0 in tl.range(0, D1, BLOCK_I):
    offs_i = i0 + tl.arange(0, BLOCK_I)
    x = tl.load(
        X + pid_b * stride_xb + offs_i[:, None] * stride_xi + offs_j[None, :] * stride_xj,
        mask=(offs_i[:, None] < D1) & (offs_j[None, :] < D2),
        other=float("inf"),
    )
    best = tl.minimum(best, tl.min(x, axis=0))
tl.store(Y + pid_b * stride_yb + offs_j * stride_yj, best, mask=offs_j < D2)
```

例子：`parallel=[n, co, oh, ow]`、`reduction=[ci, kh, kw]`，可把 `(n,oh,ow)` 展平成 M，`co` 展平成 N，`(ci,kh,kw)` 展平成 K。
```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = k0 + tl.arange(0, BLOCK_K)

# Decode offs_m -> n/oh/ow, decode offs_k -> ci/kh/kw.
# A uses position x reduction: (BLOCK_M, BLOCK_K)
# B uses reduction x channel: (BLOCK_K, BLOCK_N)
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
acc += tl.dot(a, b)
```

### 从 `boundary_treatment` 生成 mask

- `boundary_treatment` 决定哪些 lane 是有效贡献。padding、输入反推越界、ConvTranspose 的 stride 整除条件、group/channel 合法性、tail M/N/K 都应组合进 load/store mask。
  - “无特殊边界 / valid convolution / all indices valid”：只需要 tile tail mask，例如 `offs < N`、`offs_m < M_TOTAL`、`offs_n < N_TOTAL`。例：`mask = offs < N; x = tl.load(X + offs, mask=mask, other=0.0)`。
  - “zero padding / 超出输入范围视为 0”：把输入反解坐标合法性合进 load mask，`other=0.0`。例：`mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)`。
  - “max pooling padding 为 -inf”：把窗口坐标合法性合进 load mask，`other=-float("inf")`；avg pool zero padding 通常 `other=0.0`，再按 `count_include_pad` 语义决定分母。
  - “ConvTranspose 逆映射 / divisible by stride / out-of-domain contributes zero”：先算 `numerator = out_pos + padding - kernel_offset * dilation`，再把 `numerator % stride == 0` 和反解输入坐标 bounds 同时合进 mask，`other=0.0`。例：`valid = (num_h % STRIDE_H == 0) & (0 <= ih) & (ih < H_IN)`。
  - “group/channel 合法性”：把 group 内通道关系合进 weight/input mask，例如 `co // OUT_PER_GROUP == ci // IN_PER_GROUP`；不同 group 之间贡献为 0。
  - “triangular / tril / triu / clamp / Hardtanh”：这是输出值域或结构 mask，不是越界 load mask；先保证 load mask 安全，再用 `tl.where(condition, value, 0 or clipped_value)` 处理，store 仍要带 tail mask。
  - “softmax no mask / attention no causal mask”：语义上没有 causal/padding mask，但块内 softmax 仍要用 `offs_j < S` 屏蔽 tile tail，并把无效 lane 置为 `-inf`。
- 无效 load 的 `other` 要匹配 reduction 语义：
  - sum/dot 用 `0.0`。
  - min/argmin 用 `+inf`。
  - max/argmax 用 `-inf`。
  - argmin/argmax tie-breaking 规则：归约 tail 用 `+inf/-inf` 屏蔽；相等时保持 PyTorch 的 first-index 语义，不要只比较值丢掉索引。
- mask 必须在 load 前阻止越界地址参与内存访问，不能先越界 load 再用 `tl.where` 修正数值。

## 3. `tl.dot` lowering 判断

优先使用 `tl.dot` 的条件：
- 公式包含乘法加和 contraction：`sum_k lhs(...) * rhs(...)`。
- 一个 program 能覆盖真实的多个 M 位置和多个 N 通道/feature，而不是只算单个标量。
- K 可以展平成一维 `K_total`。K_total 不是 16 的倍数也可以使用 `tl.dot`，用 padded `BLOCK_K` 和 mask 过滤尾部。
- 输出 tile 能复用输入或权重 tile，例如多个空间位置共享同一组 weight，或多个输出通道共享同一组 input patch。

不应使用 `tl.dot` 的场景：
- reduction 是 max/min/mean/variance，但没有两个输入的乘法 contraction。
- depthwise conv/grouped conv 中每个输出通道只依赖极少输入通道，不能形成有效 dense `(BLOCK_K, BLOCK_N)` weight tile；除非 group 内能被组织成 dense 小 GEMM。
- K 很小且 N 也很小、`tl.dot` 资源开销大于收益时，可用多输出 tile + `tl.static_range` outer-product，但只适合很小的编译期展开：展开次数建议 `<=32`，`32~64` 仅在循环体很简单且收益明确时使用；若可能超过 `64` 应优先改用 `tl.dot`、`tl.range` 分块归约或两阶段 reduction，可能超过 `100` 时不要用 `tl.static_range` 全展开。即使用 outer-product，也仍要覆盖多个输出位置/通道，不能退化成一输出一 program 的标量循环。

## 4. Tile 语义与静态信息

- tile 大小首先由 MathIR 语义决定：一个 program 应覆盖真实的多个输出位置/通道，或覆盖一个有意义的 producer tile；具体 API 对 tile bound、shape 和精度的限制见 API 文档。
- `K_total`、`WINDOW`、`BLOCK_POS` 等派生静态大小来自 `symbol_binding`、`axis_mapping` 和任务 shape，可作为 schedule 参数；动态真实边界仍通过 mask 保持语义正确。
- 如果二维 grid 展平成一维，线性 pid 解码必须匹配快变维：通常 `pid_m = pid // blocks_n`、`pid_n = pid % blocks_n`。优先使用原生 2D grid。
- `BLOCK_M` 必须映射到真实的多个输出位置或 producer 位置。不要在一最终输出一 program 内调用很多次 `tl.dot((1, BLOCK_K), ...)` 来假装矩阵化。

## 5. 静态轴与动态边界

MathIR 中 `axis_mapping` 的 `binding=static` 和 `value` 可以作为 lowering 依据，但静态轴只说明语义范围可被 schedule 固化，不代表可以忽略动态 verifier case 的真实边界。

- 对 Conv/ConvTranspose：stride、padding、dilation 会影响反推输入坐标和合法 lane，必须进入 mask 逻辑。
- 对动态或 verifier 多 case shape：schedule 可以使用静态上限或 padded tile，但循环体内要用真实 `K_total`、输入坐标合法性和 store tail mask 跳过无效 lane。
- `tl.static_range` 会在编译期展开循环体，适合固定很小窗口或很小 K 轴（经验上 `<=32` 次展开最稳，`32~64` 谨慎）。如果展开次数可能超过 `64`，尤其可能超过 `100`，不要生成长串 `static_range`；改用 runtime `tl.range`、`tl.dot` K tile 或两阶段归约。
- 对 large reduction：不要用一输出一 program 的长串行循环。让一个 program 覆盖多个输出列/位置，或使用两阶段 reduction。
- 对反卷积 inverse mapping：先算 numerator，再分别做整除 mask 和 bounds mask，例如 `(num_h % stride_h == 0) & (0 <= ih) & (ih < H_IN)`；所有无效 input load 用 `other=0.0`。

## 6. MathIR 示例展示

### Matmul / Linear

- 对 `Y[m,n] = sum_k A[m,k] * B[k,n]`，令 A tile 为 `(BLOCK_M, BLOCK_K)`，B tile 为 `(BLOCK_K, BLOCK_N)`，accumulator 为 `(BLOCK_M, BLOCK_N)`。
- PyTorch `nn.Linear` 的原始 weight 布局是 `(O, I)`。计算 `X(N,I) @ W^T` 时，B tile 形状应是 `(BLOCK_K, BLOCK_O)`，取数地址按 `W[o, k]`：`W_ptr + offs_o[None, :] * stride_o + offs_k[:, None] * stride_i`。
- 如果需要预打包 weight，应在 `ModelNew.__init__` 中一次性完成，并保存为 parameter/buffer；`forward()` 不要反复 `permute/contiguous`。

### Conv2d / Conv3d implicit-GEMM

Conv2d 公式通常为：
`Y[n,co,oh,ow] = bias[co] + sum_{ci,kh,kw} X[n,ci,oh*stride_h+kh*dilation_h-pad_h,ow*stride_w+kw*dilation_w-pad_w] * W[co,ci,kh,kw]`

Lowering 方式：
- 将输出位置 `(n, oh, ow)` 展平成 M：`m = n * H_out * W_out + oh * W_out + ow`。
- 将输出通道 `co` 作为 N。
- 将 `(ci, kh, kw)` 展平成 K：`k = (ci * K_h + kh) * K_w + kw`，`K_total = C_in * K_h * K_w`。
- A tile 是 implicit im2col，不需要真正 materialize。用 `(offs_m[:, None], offs_k[None, :])` 解码出 `n/oh/ow/ci/kh/kw` 后从 input 中 masked load，形状为 `(BLOCK_M, BLOCK_K)`。
- B tile 从 weight 中 load，形状为 `(BLOCK_K, BLOCK_N)`。K 相关字段必须在行方向 `[:, None]`，输出通道字段必须在列方向 `[None, :]`。
- `acc += tl.dot(a, b)` 后再融合 bias、activation、scale、clamp 等 cheap epilogue。
- store tile 形状为 `(BLOCK_M, BLOCK_N)`，位置字段使用 `[:, None]`，通道字段使用 `[None, :]`，store mask 写成 `mask_m[:, None] & mask_n[None, :]`。

例子：将 MathIR Conv2d 表达式 lower 到 `tl.dot`。

MathIR 可能给出如下公式：
`Y[n,co,oh,ow] = tanh(bias[co] + sum_{ci,kh,kw} X[n,ci,oh+kh,ow+kw] * W[co,ci,kh,kw])`

它不是普通三重循环逐点计算，而是可以被看成矩阵乘：
- M 轴：所有输出位置 `(n, oh, ow)`，`M_total = N_batch * H_out * W_out`。
- N 轴：输出通道 `co`，`N_total = C_out`。
- K 轴：卷积归约 `(ci, kh, kw)`，`K_total = C_in * K_h * K_w`。
- A 矩阵：隐式 im2col 后的输入 patch，逻辑形状 `(M_total, K_total)`，不要真的 materialize。
- B 矩阵：权重矩阵，逻辑形状 `(K_total, C_out)`，从原始 `W[co,ci,kh,kw]` 按 `(k, co)` 方式读取。
- C 矩阵：输出 tile，逻辑形状 `(M_total, C_out)`，再 reshape 回 `(N_batch, C_out, H_out, W_out)`。

Triton kernel 中的核心索引骨架：
```python
# pid_m covers BLOCK_M output positions; pid_n covers BLOCK_N output channels.
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

n = offs_m // (H_OUT * W_OUT)
spatial = offs_m % (H_OUT * W_OUT)
oh = spatial // W_OUT
ow = spatial % W_OUT

acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k0 in range(0, K_TOTAL_PADDED, BLOCK_K):
    offs_k = k0 + tl.arange(0, BLOCK_K)
    ci = offs_k // (K_H * K_W)
    rem_k = offs_k % (K_H * K_W)
    kh = rem_k // K_W
    kw = rem_k % K_W

    ih = oh[:, None] * STRIDE_H + kh[None, :] * DILATION_H - PAD_H
    iw = ow[:, None] * STRIDE_W + kw[None, :] * DILATION_W - PAD_W

    a_ptrs = X + n[:, None] * stride_xn + ci[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
    a_mask = (
        (offs_m[:, None] < M_TOTAL)
        & (offs_k[None, :] < K_TOTAL)
        & (ih >= 0) & (ih < H_IN)
        & (iw >= 0) & (iw < W_IN)
    )
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = (
        W
        + offs_n[None, :] * stride_wco
        + ci[:, None] * stride_wci
        + kh[:, None] * stride_wkh
        + kw[:, None] * stride_wkw
    )
    b_mask = (offs_k[:, None] < K_TOTAL) & (offs_n[None, :] < C_OUT)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc += tl.dot(a, b)

bias_val = tl.load(bias + offs_n, mask=offs_n < C_OUT, other=0.0)
acc += bias_val[None, :]
out = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0  # tanh epilogue when API docs do not expose tl.tanh

out_ptrs = Y + n[:, None] * stride_yn + offs_n[None, :] * stride_yc + oh[:, None] * stride_yh + ow[:, None] * stride_yw
out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < C_OUT)
tl.store(out_ptrs, out, mask=out_mask)
```

这个例子中的关键点：
- `a` 必须是 `(BLOCK_M, BLOCK_K)`，位置轴来自 `offs_m[:, None]`，归约轴来自 `offs_k[None, :]`。
- `b` 必须是 `(BLOCK_K, BLOCK_N)`，归约轴来自 `offs_k[:, None]`，通道轴来自 `offs_n[None, :]`。
- `BLOCK_M` 表示多个真实输出位置，`BLOCK_N` 表示多个输出通道；这样 `tl.dot` 才能调用矩阵计算单元并复用 input/weight tile。
- padding、dilation、stride、tail K 都放进 mask；不能越界读取 input 后再依赖数值修正。
- bias、tanh、scale、clamp 等 cheap epilogue 应在 `acc` 上融合，避免额外 kernel 或中间 tensor。

1x1 Conv 是更简单的 GEMM：
- M = `N * H * W`，N = `C_out`，K = `C_in`。
- A 从 input 的 `(n,ci,h,w)` 取值，B 从 weight 的 `(co,ci,0,0)` 取值。

ConvTranspose 需要从输出位置反推输入位置：
- 对每个空间维：`num = out_pos + padding - kernel_offset * dilation`，只有 `num % stride == 0` 且 `in_pos = num // stride` 落在输入范围内时有效；无效贡献必须通过 load mask 变为 0。
- PyTorch `nn.ConvTranspose*d.weight` 原始布局是 `(C_in, C_out/groups, *kernel)`。按公式直接读取 `W[ci, co_g, k...]`；不要额外使用 `K-1-k` 翻转原始权重，除非 PyTorch 源码显式翻转了权重。
- `output_padding` 只影响输出 shape 和尾部输出位置是否存在，不改变反推输入坐标公式。
- 仍可展平成 contraction，但 mask 必须包含反推索引合法性；不确定时先生成正确的 tiled producer，再考虑预打包或 `tl.dot`。

### Pool / Norm / Activation 融合

- Conv/GEMM 是 heavy producer，bias、scale、activation、hard clamp、小窗口 pool、small fixed reduction 是 cheap consumer，应优先在寄存器中融合。
- 对 Conv -> small pool -> activation，常用方式是让 `BLOCK_M` 个最终输出位置展开成 `BLOCK_POS = BLOCK_M * pool_window` 个 producer 位置，先用 `tl.dot((BLOCK_POS, BLOCK_K), (BLOCK_K, BLOCK_N))` 生成 producer tile，再 reshape 成 `(BLOCK_M, pool_window, BLOCK_N)` 做 pool。
- 对 GroupNorm / LayerNorm / RMSNorm，重点是组内统计 reduction。它们通常不是 `tl.dot` contraction；如果前面有 Conv/GEMM producer，可以考虑 producer tile 后在寄存器里做 group 内 `sum/sum_sq/max`，否则用专门的 reduction kernel。
- Cheap consumer 不应重复计算 heavy producer。如果全融合太复杂，可以分阶段：第一个 Triton kernel 写出 heavy producer，第二个 Triton kernel 处理 cheap post-op。

## 7. 常见错误检查

- A tile `(BLOCK_M/BLOCK_POS, BLOCK_K)` 中，位置字段用 `[:, None]`，K 字段用 `[None, :]`。
- B tile `(BLOCK_K, BLOCK_N)` 中，K 字段用 `[:, None]`，N/channel 字段用 `[None, :]`。
- `acc += A * B` 不能替代 `tl.dot(A, B)`，因为它会混淆 K 轴和输出通道轴。
