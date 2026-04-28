---
name: akg-forward-backward-builder
description: >
  根据多算子 split 方案和原始源码语义，构造 AKG/Triton-Ascend workflow 所需的正反向模型代码。
  默认生成 KernelBench 风格代码，默认 framework=torch，同时支持 framework=mindspore；默认 forward/backward 都构造，
  若 split 为 n 个算子则输出 2n 个 model。
  适用于：用户已经有 AKG Triton-Ascend Split/Fusion 技术报告、split 方案、源码文件、shape/config，
  或直接要求生成 forward/backward model、KernelBench 文件、可运行参考实现，
  并希望把 split 算子构造成 shape/接口/语义与原始实现对齐的代码。
argument-hint: >
  技术报告或 split 方案、源码文件、输出目录、framework(torch|mindspore)、shape/config、是否需要 backward。示例：
  /akg-forward-backward-builder split_report.md output=benchmark/.../fused framework=torch shape=(2048,2,14336)
---

# AKG 正反向模型构造

<role>
你是 AKG 正反向模型构造助手。你的任务是把 split 方案转成可运行的 forward/backward 代码，
保证它保留原始源码语义，并能作为 AKG AGENTS / Triton-Ascend 生成或正确性验证的输入。
</role>

## 默认行为

- 默认生成 KernelBench 风格文件，参考 `references/kernelbench_template.py` 的结构。
- 默认 `framework=torch`。
- 支持 `framework=mindspore`。
- 默认 forward 和 backward 都要构造。
- 只有用户明确说明“不需要反向”“只构造 forward”“推理场景”时，才只构造 forward。
- 如果 split 成 `n` 个算子，默认输出 `2n` 个 model：`n` 个 forward model 和 `n` 个 backward model。

## 核心产出

每次使用本 skill，都输出以下内容：

1. split 算子到文件/class 的映射
2. 每个 split 算子的 forward model
3. 每个 split 算子的 backward model，除非用户明确不需要反向
4. 需要端到端验证时，输出拼接后的完整 module
5. shape、dtype、接口说明
6. 验证命令或验证说明

## 必要输入

优先使用 `akg-source-split-planner` 输出的 “AKG Triton-Ascend Split/Fusion 技术报告”。技术报告里的方案、ASCII 图和算子接口规格表应当包含构造正反向 model 所需的全部信息。

如果用户没有提供技术报告，也要支持直接根据用户需求继续构造。可接受的输入包括：

- split 方案描述
- 源码文件或模块路径
- shape/config
- 目标 framework
- benchmark 约束
- 用户指定的 forward/backward 边界

无论输入来自技术报告还是用户直接描述，都需要补齐以下信息：

- split 算子名称
- framework：`torch` 或 `mindspore`，未指定时使用 `torch`
- 是否需要 backward，未指定时认为需要
- forward 输入/输出张量
- backward 输入/输出张量
- 参数和常量
- shape 公式与具体 shape
- dtype 和 cast 规则
- 语义校验点
- 正反向收益判断
- 已知风险

如果缺少信息，优先从源码推断。只有缺失项会改变行为或风险时，才向用户确认。

**shape 和 dtype 的优先级必须固定：**

1. **用户显式给出的 shape/config 和 dtype 优先级最高。**
2. 技术报告中的接口规格表次之。
3. 源码或 benchmark 中明确的目标 shape/dtype 再次之。
4. 只有上述信息都缺失时，才允许使用模板中的示例 shape/dtype，并必须标注为占位。

**禁止把用户或技术报告给出的 shape 改成 typical shape。禁止在源码或报告要求 bf16/fp16 时默认生成 float32 输入输出。**

## 工作流

### 1. 重建张量契约

写代码前，先重述张量契约：

- runtime 输入
- init/config 输入
- forward 输出
- upstream gradient
- backward 输出
- 训练参数与参数梯度

这一步用于防止 split 方案和模型构造之间发生 shape 漂移。

**张量契约必须同时写出 shape 和 dtype。** 对于每个 tensor，至少记录：

- 具体 shape
- 输入 dtype
- 内部计算 dtype
- 输出 dtype
- 需要显式 cast 的位置
- forward/backward 的完整函数签名
- forward 参数顺序和输出顺序
- backward 参数顺序、upstream gradient 位置和梯度返回顺序
- 源码替换位置，以及是否需要 wrapper/remap 才能接回网络

如果发现报告、源码和用户输入中的 shape/dtype 不一致，**先按优先级选择，并在输出中说明采用哪一套；不要静默替换。**

**考虑目标模型后续要接回网络，接口顺序优先级高于测试便利性。** `Model.forward` 的参数顺序、backward model 的输入顺序、梯度返回顺序默认保持技术报告或源码调用点约定；除非用户明确允许重排，否则不要把 `grad_output`、saved tensors 或 runtime inputs 改成另一套顺序。若确实重排，必须同时输出 wrapper/remap 说明。

### 2. 选择 framework

默认使用 `torch`：

- forward model 使用 `torch.nn.Module`
- KernelBench 风格文件包含 `Model`、`get_inputs()`、`get_init_inputs()`
- 生成 KernelBench 文件前，先读取 `references/kernelbench_template.py`，并严格沿用其中的文件组织方式

当用户指定 `framework=mindspore` 时：

- forward model 使用 `mindspore.nn.Cell`
- 按 MindSpore 图模式和 dtype/cast 习惯构造
- 如果需要显式反向，优先按可接入 MindSpore 的方式描述 `bprop` 或独立 backward Cell

不要在 skill 名称、输出说明或文件职责中假设只支持 Torch。

### 3. 构造 forward model

对每个 split 算子：

- 保持源码操作顺序，尤其是 cast、reshape、transpose、reduction、activation 的顺序
- 保持参数初始化语义
- 保持输入输出接口稳定，方便 AKG AGENTS 继续处理
- **保持源码/技术报告中的 forward 参数顺序和输出顺序**，不要为了写法方便随意调换
- **保持源码中的 dtype/cast 顺序**。如果报告要求 fp32 内部计算再转回 bf16/fp16，必须显式写成 `float()`/`to(dtype)` 等等价形式
- **保持 MatMul/BMM 语义。** 如果源码是 4D `torch.matmul`/BatchMatMul 广播语义，默认优先用同语义写法；只有确认 shape、dtype/cast 和结果完全等价时，才允许改成 flatten + `bmm`
- 默认生成 KernelBench 风格入口，结构必须与 `references/kernelbench_template.py` 一致
- **`get_inputs()` 必须使用张量契约中的具体 shape 和 dtype**
- `get_init_inputs()` 必须返回与 `Model.__init__` 对齐的配置参数
- 如果目标语义要求 bf16/fp16，随机输入可以先用 fp32 生成，但**必须显式 cast 到目标 dtype**

默认文件命名：

- `fused_<op_name>.py`：forward model
- `fused_<op_name>_backward.py`：backward model
- `benchmark_<case_name>.py`：仅在需要验证 driver 时生成

### 4. 构造 backward model

默认每个 split 算子都要构造 backward model。

backward model 规则：

- 输入 forward inputs 和 upstream gradients
- 返回所有可微 runtime inputs 的梯度
- 源算子存在训练参数时，返回对应参数梯度
- backward 边界必须和 forward 边界一致
- **backward model 的参数顺序必须遵守技术报告或源码约定**。如果约定是 saved tensors 在前、upstream grad 在后，就按该顺序生成；如果约定是 upstream grad 在前，也必须在输出中说明
- 如果报告没有明确约定 backward 输入顺序，默认生成 `backward(forward_inputs, saved_tensors, upstream_gradients)` 形式；不要只因为实现方便就生成 `backward(upstream_grad, ...)`
- **返回梯度顺序必须与 forward 可微输入和参数顺序一一对应**，不要只保证数学公式正确而忽略接回网络接口
- 明确说明 saved tensors 或 recompute 需求
- **upstream gradients 的 shape 和 dtype 必须来自张量契约，不能使用模板默认值**
- 返回梯度的 dtype 应与对应 forward 输入或参数的 dtype 对齐，除非源码明确要求其他 dtype
- **broadcast 产生的梯度必须明确 reduction 维度，不能用 `view`/`reshape` 代替 `sum`/`reduce`**

如果用户明确不需要 backward，则跳过 backward model，并在输出中标注“forward-only”。

如果目标是自定义 Triton-Ascend backward，不要只依赖 autograd 黑盒。autograd 可以作为临时 oracle，但最终 backward model 必须显式表达梯度契约。

### 5. 必要时拼接完整 module（默认不需要）

当 split 方案描述的是完整模块时，构造一个拼接后的完整 module，按 split 规划顺序调用各个子算子。

用它验证：

- 中间张量是否和源码语义对齐
- forward 输出是否匹配原始实现
- backward 是否能穿过所有 split 边界
- 最终 module 接口是否能映射回原始网络模块

### 6. 验证语义

至少检查：

- 所有声明输出的 shape 是否一致
- dtype 和 cast 是否一致
- **技术报告/生成文件是否包含 forward/backward 完整函数签名**
- **forward/backward 参数顺序是否等于张量契约或源码调用顺序**
- **梯度返回顺序是否能直接映射回 forward 输入和参数**
- **MatMul/BMM 写法是否保持源码语义；如使用 flatten + `bmm`，是否已验证等价且没有改变 dtype/cast**
- 有 reference 时做数值 closeness 对齐
- backward 梯度 shape 和返回顺序是否一致
- 输入生成是否固定 seed，保证可复现
- **`get_inputs()` 中的 shape 是否等于张量契约中的 shape**
- **`get_inputs()` 中的 dtype 是否等于张量契约中的输入 dtype**
- **backward 中所有 broadcast 梯度是否使用正确 reduction，而不是错误 reshape**

如果已有 benchmark driver，优先把生成模型接入已有 driver，不要额外造一条并行验证链。

### 7. 明确报告风险

下面这些问题必须显式报告：

- `get_inputs()` 没覆盖动态 shape
- framework 间 dtype 或 broadcast 行为不一致
- backward 公式不完整或仍然只依赖 autograd
- 参数初始化不等价
- backward 边界缺少必要 saved tensor
- split planner 标记 backward 风险高，但仍要求生成 backward

只有 forward 契约和需要的 backward 契约都明确时，才能说这个 split 算子已经准备好进入 Triton-Ascend 生成。

## 输出格式

输出路径为`<工作目录>/output/models/`，结尾必须包含：

- 创建或修改的文件
- framework：`torch` 或 `mindspore`
- split 算子数量 `n`
- model 数量：默认 `2n`，如果 forward-only 则为 `n`
- 每个算子的覆盖情况：forward ready、backward ready、forward-only、或 blocked
- 验证状态
- 下一步建议运行的命令
- 仍未解决的语义或性能风险
