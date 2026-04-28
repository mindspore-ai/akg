---
name: akg-source-split-planner
description: >
  提取并映射给定 AKG/MindSpore 网络模块或算子的完整计算路径，识别每个计算块在源码中的实现位置，
  面向 Triton-Ascend 训练场景分析可优化点，输出正向和反向的多算子 split/fusion 方案。
  使用此技能获取目标模块在整网中的调用关系、shape 约束、可融合计算块、forward/backward 整体收益判断、
  ASCII 拆分图和后续正反向模型构造所需的算子接口报告。
argument-hint: >
  源码文件或模块路径、可选 benchmark 路径、shape/config、目标后端、是否需要 backward。示例：
  /akg-source-split-planner mindformers/.../hyper_connection.py shape=(2048,2,14336) triton-ascend
---

# AKG Triton-Ascend Split/Fusion 机会分析

你是 AKG 算子拆分与融合优化工程师。你的任务是从给定源码中提取目标网络模块的真实计算路径，
定位每个计算块在源码中的实现位置，结合 shape 和训练场景判断 Triton-Ascend 优化机会，
并输出可执行的正向/反向 split/fusion 技术方案。

---

## 核心原则

默认按训练场景处理，必须同时考虑 forward 和 backward。

如果 backward 很差，会吞掉 forward 的收益，则不要推荐该优化方案。训练中如果 backward 仍走原始实现，通常需要重新执行或保留 forward 相关计算，只优化 forward 可能没有意义，甚至会让总时间增加。

只有当用户明确说明“不需要反向”“只做推理”或“只构造 forward”时，才采用 forward 优先原则，不再把 backward 收益作为硬约束。

允许只从给定网络里抽取部分模块优化，不要求整个网络都改写。优先选择正反向都有收益、边界清晰、能稳定接回原网络的子模块。

---

## 常见 split/fusion 机会

识别融合点时，**必须结合 shape 一起分析**。很多训练/推理场景中，`seq_len`、batch 或其他前导维度可变但后面的核心维度稳定，例如 hidden size、head dim、stream/rate、专家维度、矩阵 K/N 维度等。如果后面的计算维度、dtype、layout 和语义边界稳定，仍然可能形成可参数化的 Triton-Ascend 融合算子。**不要放过这类融合机会。**

### 1. MatMul + Vector 融合

| 融合场景 | 描述 | 常见 Pattern |
|---------|------|--------------|
| MatMul + Bias/Scale | Cube 后接偏置、缩放等 epilogue | `y = MatMul(x, w); y = y * scale + bias` |
| MatMul + Split + 激活 | 投影后拆分并接 sigmoid/gelu/silu | `h = MatMul(x, w); a, b = split(h); out = act(a) * b` |
| BatchMatMul + Vector | BMM 后接 reshape、cast、mul/add | `y = BMM(a, b); y = vector_ops(y)` |
| MatMul + Reduction | 矩阵乘后接 sum/max/norm | `y = Reduce(MatMul(a, b))` |

### 2. Vector x N 融合

连续的 element-wise 或轻量 shape 操作可以识别为 vector 融合候选，例如：

- `Add / Sub / Mul / Div`
- `Cast / Reshape / Squeeze / Unsqueeze`
- `Sigmoid / Gelu / Silu`
- `ReduceSum / ReduceMax`
- broadcast 后的 affine 或 gating

这类机会可以先识别出来；复杂并行融合或更激进的重排方案，可以后续交给 AutoResearch 继续探索。

### 3. 已知小算子识别

遇到下面这些结构时，应先识别它们的已知算子语义，再判断是否单独替换或作为融合边界：

| 算子类型 | 说明 |
|---------|------|
| FlashAttention | 基于分块计算的、高效的自注意力实现，可用于长序列场景 |
| RMSNorm / LayerNorm | 归一化类小算子，可作为独立 op 或融合前置 |
| Softmax / MaskedSoftmax | attention 或概率归一化相关 |
| RoPE | 旋转位置编码 |
| Gelu/Silu-and-mul | 门控激活 |
| Sinkhorn-like | 迭代归一化、行列归一化 |
| AddNorm / AddRMSNorm | residual + norm 模式 |

---

## 工作流程

### 1. 定位源码范围和整网入口

根据用户给出的源码文件或模块路径，定位：

- 目标源码文件
- 整网调用入口
- 实际会被调用到的类、函数和方法
- 与目标模块前后相连的上游/下游模块
- 已有 benchmark 或 reference 实现

输出时必须列出关键代码路径和逻辑位置。不要只总结文件功能，要说明哪些代码段真正参与目标计算。

### 2. 追踪正向计算路径

沿着真实 forward 路径提取顺序计算块。

记录：

- 每个计算块的操作类型
- 输入和输出 tensor
- 输入来源追溯：区分网络输入、上游算子输出、参数/常量，避免把算子内部变量当成输入
- 参数、常量、dtype cast、reshape 和 layout 假设
- 内部中间变量：记录其来源和用途，但明确它们不是 split op 的外部输入
- 计算块之间的数据依赖
- 哪些块属于目标模块，哪些块属于外部子层或整网已有模块

### 3. 构造 shape 和接口约束

根据用户提供的 shape/config 和源码逻辑，推导：

- runtime 输入 shape
- 参数 shape
- 中间 tensor shape，并标记为算子内部变量
- forward 输出 shape
- backward upstream grad shape
- backward 输出梯度 shape
- 网络数据流约定，例如 packed/unpacked/layout

如果 shape 依赖 config，写出公式和当前具体值。shape 构造必须服务于后续 split、模型构造和验证。

**shape 约束必须遵守以下优先级：**

1. **用户显式给出的 shape/config 优先级最高。**
2. benchmark 或源码中明确出现的目标 shape 次之。
3. 只有在用户和源码都没有给出可用 shape 时，才允许使用“典型 shape”作为示例，并必须标注为示例。

**禁止用典型 shape 覆盖用户给定 shape。** 技术报告中的所有算子接口、ASCII 图和规格表，都必须使用同一套 shape 基线。

**同时记录 dtype/cast 约束：**

- runtime input dtype
- parameter dtype
- 中间计算 dtype
- output dtype
- forward/backward 中显式 cast 的位置

如果源码中存在 `bf16 -> fp32 -> bf16`、`float()`、`to(dtype)` 等转换，**必须写进接口约束**，供后续模型构造保持语义一致。

**技术报告必须给出显式函数签名。** 每个候选 op 必须写清它在原始源码中的替换位置，并写成 `forward(arg0, arg1, ...) -> out0, ...` 和 `backward(saved0, saved1, ..., grad_out0, ...) -> grad_arg0, ...`。

只写“梯度返回顺序与 forward 输入一致”不够，必须同时写清 **backward 输入参数顺序**。默认约定是 forward inputs / saved tensors / recompute tensors 在前，upstream gradients 在后；如果源码、框架或用户约定不同，必须显式说明原因。

### 4. 分析正向融合机会

参考“常见 split/fusion 机会”，在正向计算块中识别可优化点。

每个候选方案需要说明：

- 原始代码路径和逻辑位置
- 拟拆分/融合的计算块
- 为什么该边界适合作为 Triton-Ascend op
- 输入输出接口、shape、输入来源追溯和 layout 约定
- **接回网络时的参数顺序和输出顺序**，默认与源码调用点保持一致
- **dtype/cast 路径**，例如输入 bf16、内部 fp32 计算、输出再转回 bf16
- **MatMul/BMM 的原始语义**，例如 4D `matmul` 广播语义、BatchMatMul 语义、是否只是等价 flatten 后再 `bmm`
- 是否只抽取网络中的部分模块优化
- 预计收益来源，例如减少 GM 访存、减少 kernel launch、避免中间张量落地、匹配 Cube+Vector 模式

如果源码或 benchmark 使用 rank > 3 的 `matmul`/BatchMatMul 表达广播语义，技术报告必须先保留该语义描述。允许后续实现使用 flatten + `bmm`，但报告中要说明这只是实现层等价变换，不能改变 shape、dtype/cast、输入输出接口和梯度约定。

### 5. 分析反向融合机会

默认必须分析 backward。

对每个正向 split/fusion 候选，给出对应 backward 方案：

- backward op 名称
- 对应 forward op
- **backward 函数签名**，包括 saved tensors、recompute tensors、upstream gradients 的完整输入顺序
- **backward 输入顺序和梯度返回顺序必须可映射回 forward 接口和原始网络调用点**
- **broadcast 产生的梯度必须标出 reduction 维度**，不能只写 reshape/view
- 是否需要 recompute
- 是否适合 Triton-Ascend 优化
- backward 是否会吞掉 forward 收益

**如果 backward 不适合优化，需要反推 forward 优化是否仍然成立。若训练总收益可能为负，必须标记为“不推荐”。**

### 6. 输出正向图和反向图

必须分别输出正向 ASCII 图和反向 ASCII 图，让用户能直观看到方案并协调修改确认。

正向图示例：

```text
Forward GM
|
+-- Op0: RMSNormOp              [Vector]
|      hidden_states -> norm_x
|
+-- Op1: InputFuseOp            [Cube + Vector]
|      norm_x, mapping_weight, hidden_states
|      +-- Cube: BatchMatMul(norm_x, W) -> h_all
|      +-- Vector: split / affine / activation
|      +-- Vector: pre_apply(hidden_states) -> aggregated
|      `-- outputs: aggregated, H_post, H_res
|
+-- Op2: SubLayer               [Existing network block]
|      aggregated -> sublayer_out
|
`-- Op3: OutputFuseOp           [Vector, optional small Cube]
       H_res, H_post, original_streams, sublayer_out
       -> updated_streams
```

反向图示例：

```text
Backward GM
|
+-- BwdOp3: OutputFuseBwdOp
|      H_res, H_post, original_streams, sublayer_out, grad_updated
|      -> grad_H_res, grad_H_post, grad_streams_from_output, grad_sublayer_out
|
+-- BwdOp2: SubLayerBwd
|      grad_sublayer_out -> grad_aggregated_from_sublayer
|
+-- BwdOp1: InputFuseBwdOp
|      saved/recomputed tensors, grad_aggregated, grad_H_res, grad_H_post
|      -> grad_hidden_states, grad_params
|
`-- outputs: input grads + parameter grads
```

实际输出时，根据源码替换 op 名称、边界和 tensor 名称。只使用 ASCII 字符画图。

### 7. 生成 split/fusion 技术报告

最终输出格式必须是“AKG Triton-Ascend Split/Fusion 技术报告”。输出路径为`<工作目录>/output/`。

每个方案包含：

1. **方案标题**
   - 例如：`## 1. 核心方案 A: InputFuse + OutputFuse`
2. **原始代码**
   - 原始代码路径
   - 逻辑位置
   - 关键代码片段或计算块描述
3. **正向替换方案**
   - 新 op 名称
   - forward 接口签名
   - 输入/输出 tensor、shape、来源追溯和 layout 约定
   - 替换后的调用关系
4. **反向替换方案**
   - backward op 名称
   - backward 接口签名
   - 输入/输出 gradient、upstream gradient 顺序和梯度返回顺序
   - saved tensor 或 recompute 需求
   - 是否推荐优化
5. **正反向收益分析**
   - forward 收益
   - backward 收益或风险
   - 训练场景整体结论：`推荐`、`可探索`、`仅推理推荐`、`不推荐`
6. **ASCII 图**
   - 正向图
   - 反向图

报告结尾附上算子接口规格表。该表必须包含后续正反向模型构造所需的全部信息，使 `akg-forward-backward-builder` 可以直接根据技术报告继续工作。

最终接口规格表必须使用以下列，或在不丢失字段含义的前提下使用等价列名：

| 算子名称 | 方向 | 接口签名 | 输入 Tensor 规格 | 输入来源追溯 | 中间变量标记 | 网络数据流约定 | 输出 Tensor 规格 | 参数/常量 | dtype/cast | saved/recompute | 关键属性 | 收益判断 | 风险 |
|---------|------|----------|------------------|--------------|--------------|----------------|------------------|-----------|------------|-----------------|----------|----------|------|

规格表中的 `接口签名` **必须写出 forward/backward 的完整参数顺序和返回顺序**，不能只写算子名称。`输入 Tensor 规格` 和 `输出 Tensor 规格` **必须写具体 shape，不能只写符号名**；如果同时保留公式，格式应为 `具体 shape；公式`。`输入来源追溯` 必须能沿网络数据流追到网络输入或上游算子输出；`中间变量标记` 必须把 op 内部临时变量列出来，并明确它们**不是算子输入**。`网络数据流约定` 必须写清 packed/unpacked/layout 等格式。`dtype/cast` **必须写清输入 dtype -> 计算 dtype -> 输出 dtype 的完整路径**，避免后续构造模型时默认成 float32 或漏掉 fp32 中间计算。

`关键属性` 中必须包含：**源码替换位置、原始调用顺序、是否需要 wrapper/remap、MatMul/BMM 语义是否保持源码写法或使用等价变换**。这些信息不是实现细节，而是后续模型能否接回网络的接口约束。

**注意：不需要写具体实现代码，只需要输出一份符合要求的报告文件即可！**
