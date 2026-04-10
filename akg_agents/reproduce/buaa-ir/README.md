# AKG Agent 正确性优化与元提示增强说明

本文档总结了基于 AKG Agent 的一轮结构性优化工作。此次修改以提升大语言模型生成 Triton 算子代码时的**正确性**为首要目标，并兼顾生成代码的**性能可优化性**与**本地调试效率**。整体思路围绕 AKG Agent 的 Designer–Coder–Verifier 闭环，对算子信息注入、Prompt 结构、调试链路和细粒度元提示机制进行系统整理，使模型在“理解算子—生成 IR—生成代码—调试修复”这一流程中获得更稳定、更可控的约束。

## 0. 基准版本与实验配置

* **commit**：`8da1e29a678f341cf2e8cd9e30dfd1c3b848a30d`
* **workflow**：`default_workflow`
* **model**：`deepseek_reasoner`
* **gpu**：`nvidia a100`
* **npu**：`ascend910b4`
* **测试集**：`KernelBench Level 1`

## 1. 背景与目标

AKG Agent 采用多智能体协同框架完成算子生成任务，其中 Designer 负责生成 Unified Sketch，Coder 负责将 Sketch 落实为目标 DSL 代码，Verifier 负责编译、正确性校验与性能评估，Conductor 负责根据报错类型进行动态调度。该架构的优势在于将“优化策略设计”与“具体代码实现”解耦，但在实际使用中，若上游 Sketch 约束不清、下游 Prompt 冗余或错误日志缺失，就会直接影响整体生成质量。

本次工作聚焦两个核心目标：

1. **提升正确性**：减少参数 shape 错误、stride 推断错误、Triton 语法误用、错误调试方向偏移等问题，提高 pass@1 的成功率。
2. **增强性能表达能力**：在不显著增加系统复杂度的前提下，引入细粒度元提示机制，使 Designer 输出的 IR 能显式表达分块、并行、访存、流水线等优化语义，为后续性能优化提供统一接口。

## 2. 问题描述

在基准版本的默认生成流程（`Designer -> Coder -> Verifier`）中，主要暴露出以下问题：

1. **参数 shape 信息缺失或理解错误**
   大模型对部分算子的输入输出维度、权重布局、广播关系理解不准确，进而导致索引表达式和访存逻辑错误。

2. **错误使用 `data_ptr()` 等低层接口**
   在 Host 侧生成代码时，模型会将 PyTorch 张量错误地转为裸指针，既增加出错概率，也不符合当前代码框架的调用方式。

3. **Triton 语法与编程范式误用**
   常见问题包括错误使用 `continue/break`、`floordiv/floormod`、不规范的 `tl.constexpr` 参数，以及使用 Python 风格 `if` / 嵌套 `for` 处理本应由 mask 完成的 element-wise 写回。

4. **调试信息不完整，导致修复链路失真**
   原流程中错误日志存在截断现象，LLM 拿到的上下文不足，无法准确定位编译失败、运行时报错或数值错误的真正根因。

5. **Prompt 冗余，噪声较大**
   原有 Prompt 中存在较多示例性、背景性内容，压缩了真正关键的信息密度，使模型在长上下文中更容易偏离核心任务。

6. **性能提示缺失或表达粒度过粗**
   即使代码能够正确生成，也经常停留在“功能正确但实现粗糙”的水平，无法显式体现分块、访存、流水线等结构性优化意图。

## 3. 问题分析

上述问题本质上并非孤立 bug，而是由“**信息表达不充分**”与“**约束传递不稳定**”共同造成的，具体可归纳为以下三个层面。

### 3.1 Designer 侧：算子语义约束不足

Designer 的职责是生成 Unified Sketch，用中间层表达算子的核心计算逻辑与优化意图。若 Sketch 对输入张量形状、stride、索引边界、mask 语义、归约维度等信息表达不完整，Coder 即便语法正确，也会在实现时把错误结构“忠实翻译”到 Triton 代码中。Designer 的价值在于先表达高层优化策略，再交由 Coder 做具体实现；一旦设计阶段约束不清，后续实现质量会显著下降。

### 3.2 Coder 侧：代码生成与调试提示混用

原始 Coder Prompt 既承担“首次生成”任务，又承担“失败后修复”任务，但这两类任务对上下文的需求并不相同。首次生成更需要 API、shape、host/device 结构等基础信息；调试阶段则更依赖错误日志、上次失败代码与针对性修复建议。如果两者混杂在同一 Prompt 中，模型容易把注意力放在无关示例或低优先级提示上，降低 debug 效率。

### 3.3 系统侧：缺少细粒度优化语义接口

当前 AKG Agent 具有良好的多智能体骨架与文档驱动能力，但若缺少统一的、可组合的优化语义接口，系统就只能依靠 Prompt 中松散的自然语言描述去“暗示”模型如何优化。这种方式难以稳定表达不同优化维度，也不利于后续做程序特征感知搜索。更合理的方向是引入细粒度元提示，将优化策略从“文本建议”升级为“结构化语义单元”，在 Sketch 中显式挂载，并在 Designer / Coder 两端协同使用。

## 4. 解决方案

本次修改围绕“**补全算子信息、拆分生成与调试链路、压缩无效上下文、引入细粒度元提示**”展开。

### 4.1 添加算子信息提取模块

* **新增文件**：`extractor_torch.py`

基于 PyTorch FX 的 `symbolic_trace` 机制提取算子计算图，并结合 `FakeTensorMode` 与 `ShapeProp` 在不执行真实计算的前提下推导张量元信息，包括：

* shape
* dtype
* stride
* 输入输出关系
* 部分算子的结构属性

这些信息被注入到 Designer Prompt 与 Coder Prompt 中，使模型不再仅依赖自然语言描述猜测维度关系，而是能够直接基于结构化元信息生成代码。

### 4.2 Designer Prompt 修改

* **修改文件**：`gen_sketch.j2`

主要调整包括：

1. 增加卷积类算子的维度说明与计算公式提示；
2. 强化对索引逻辑、边界处理、mask 表达方式的约束；
3. 注入由 `extractor_torch.py` 提取的 shape / dtype / stride 等参数信息；
4. 引入元提示相关内容，使 Designer 在生成 Sketch 时显式体现优化意图。

调整后的目标不是让 Designer 直接输出复杂底层代码，而是让其更稳定地产生**结构正确、语义清晰、可被下游正确 lower 的 Sketch**。

### 4.3 Coder Prompt 修改

* **修改文件**：`codegen.j2`

主要改动如下：

1. 删去冗余示例，压缩 Prompt 长度；
2. 增加算子参数维度、dtype、stride 等结构化信息；
3. 将 Coder Prompt 拆分为两类：

   * **Coder Gen**：用于第一轮生成；
   * **Coder Debug**：用于后续失败修复。
4. 在 Debug Prompt 中，将“任务描述—上次代码—完整错误日志—专家建议”提前，降低无关上下文干扰；
5. 增加元提示相关内容，使 Coder 能够理解 Sketch 中的优化语义，并尽量以 Triton 合法范式实现。

### 4.4 专家文档与 API 文档压缩

* **修改文件**：`suggestion_docs.md`
* **修改文件**：`api.md`

通过压缩冗余内容、保留高频且关键的 Triton API / 经验建议，减少上下文长度，提升关键信息密度。

### 4.5 增加 Debug 专用文档

* **新增文件**：`suggestion_debug.md`
* **新增文件**：`api_debug.md`

这两份文档面向 Debug 阶段使用：

* `suggestion_debug.md`：针对常见编译错误、运行时错误、shape 错误、API 误用等问题整理专家修复建议；
* `api_debug.md`：提供精简版 API 文档，仅保留调试阶段最常用、最易错的内容。

这样做的目的是将“初次生成知识”和“失败修复知识”分层管理，提升 Debug Prompt 的针对性。

### 4.6 Designer / Coder 代码修改

* **修改文件**：`designer.py`
* **修改文件**：`coder.py`

在代码层增加元提示的加载、注入与传递逻辑，使元提示不再只是文档层概念，而能实际参与 Designer 输出和 Coder 实现过程。

### 4.7 添加元提示机制

* **新增文件**：`manager.py`, `search.j2`

这是本次工作中最关键的新增能力之一。元提示机制的目标不是直接替代 Designer 或 Coder，而是为系统提供一层**可组合、可选择、可注释的优化语义接口**。当前已完成基础结构，包括：

1. 元提示基本数据结构定义；
2. 元提示搜索 Prompt；
3. 元提示空间初筛逻辑；
4. 与 Designer / Coder 的基础集成。

## 5. 元提示设计

结合当前实现，元提示可理解为对优化意图的细粒度结构化描述，用来替代以往“只靠自然语言描述优化”的粗粒度方式。它面向的不是单一技巧，而是一组可复用的优化维度，例如：

* **Parallelism**：并行策略，如任务划分、线程块映射、归约并行方式；
* **Memory**：访存策略，如连续访问、缓存复用、块级加载、共享内存/寄存器倾向；
* **Execution / Pipeline**：执行结构，如多阶段流水、计算与访存重叠；
* **Constraint**：约束条件，如仅适用于某类后端、不能破坏原有接口、必须保持数值一致性。

在当前实现中，元提示主要作为 Designer Prompt 的组成部分出现，并在 Sketch 中以显式注释或标记的形式体现，从而让下游 Coder 能够感知“这里不仅是一段计算逻辑，更对应某种明确的优化意图”。

元提示模块已完成的工作包括：

* 元提示结构定义；
* 基于任务特征的初步筛选；
* 搜索 Prompt 搭建；
* 与 Designer / Coder 的基础联动。

## 6. 结果

### 6.1 GPU 平台

#### 测试结果

* **Benchmark**：正确生成算子个数 88
* **Ours**：正确生成算子个数 95

### 6.2 NPU 平台

#### 测试结果

* **Benchmark**：正确生成算子个数 69
* **Ours**：正确生成算子个数 71

## 7. 效果总结

优化感知 IR 可正确表示KernelBench level l 中的所有基础算子。KernelBench pass@l 正确性高于原 AKG Agent 结果。

## 8. 启用方法

本项目提供了一个针对批量算子执行进化式生成测试的入口脚本，便于开发者或测试人员直接运行验证流程。当前推荐使用 `KernelBench` 的 Level 1 数据集进行验证。

### 启动测试

```bash
time python -u run_test_all_kernels_evolve_passk.py \
    --folder thirdparty/KernelBench/KernelBench/level1/ \
    --max_rounds 1 \
    --start_index 1 \
    --end_index 10
```

### 参数说明

* `time`：统计整体任务运行时间；
* `-u`：实时输出日志，便于观察生成与验证过程；
* `--folder`：指定测试算子目录；
* `--max_rounds`：最大迭代轮数，设为 `1` 时更适合评估基础正确率；
* `--start_index` / `--end_index`：选择测试算子范围，便于局部快速验证。

### 说明

运行该命令后，系统会自动完成 Designer → Coder → Verifier 的生成、编译、运行与校验流程。日志、临时产物和评测结果将输出到工作区对应目录中，便于进一步分析。

---

## 9 Code Review 问题修复（P0 / P1）

本节对应 Code Review 报告（[`review.md`](./review.md)，审查对象 commit `fe19fda5`）中标记的 3 个 P0 阻断性问题与 3 个 P1 高风险问题，逐条说明修复方案与修复位置。

### 问题修复

#### P0-1 安全漏洞：`exec()` 无沙箱保护 → 已修复

**原始问题**：`extractor_torch.py` 中 `exec(code_str, ns, ns)` 直接执行用户代码，无任何沙箱保护，可执行任意系统命令。

**修复方案（两层防护）**：

- **第一层（AST 静态检查）**：新增 `_validate_code_safety(code_str)`，在 `exec()` 执行前用 `ast.parse()` + `ast.walk()` 扫描代码树，命中黑名单（`_DANGEROUS_MODULES`：`os`、`subprocess`、`sys` 等；`_DANGEROUS_CALLS`：`eval`、`__import__`、`system` 等）则直接抛 `ValueError`，阻止执行。
- **第二层（运行时沙箱）**：用 `_SAFE_BUILTINS` 白名单替换 `__builtins__`，仅放行 KernelBench 代码所需的内建函数（`range`、`len`、`isinstance` 等）；通过 `_restricted_import()` 拦截 `__import__`，仅允许 `torch`、`math`、`numpy` 等安全模块导入。`exec()` 改为 `exec(code_str, ns, ns)`，其中 `ns = {"__builtins__": _SAFE_BUILTINS, "__name__": "__kernelbench_sandbox__"}`。

#### P0-2 运行时错误：错误包名引用 → 已修复

**原始问题**：`coder.py` 中 `from ai_kernel_generator.core.extractor_torch import ...`，包名写错，导致 `ModuleNotFoundError`。

**修复方案**：改为正确包名：
```python
from akg_agents.core.extractor_torch import extract_kernelbench_shapes_dtypes
```

#### P0-3 逻辑错误：FX trace 失败时用输入 shape 冒充输出 shape → 已修复

**原始问题**：`symbolic_trace` 失败时 `graph_tensors` 为空，代码 fallback 为直接取第一个输入的 shape 作为输出 shape，对 matmul、conv、reduction 等算子完全错误。

**修复方案**：

- 新增 `_infer_output_shape_via_forward(model, real_inputs)` 函数：将输入转为 meta tensor，在 `torch.no_grad()` 下跑一次 meta forward，从输出张量中读取真实 shape，作为 `fallback_output_shape` 返回。
- 删除原来用输入 shape 冒充输出 shape 的 fallback 逻辑；`_extract_op_features()` 仅使用 `op_meta["fallback_output_shape"]`，不再回退到输入 shape。

#### P1-4 算子名称匹配过于宽泛（子串误命中）→ 已修复

**原始问题**：用 `any(op in target_str for op in compute_ops)` 做子串匹配，`"mm"` 会命中 `"comma"`，`"add"` 会命中 `"baddc"` 等。

**修复方案**：引入 token 级精确匹配机制（详见 5.5.2 节），将 `manager.py` 中所有算子语义判断替换为基于 `SEMANTIC_RULES` 词表的 `_rule_matches()` 调用，匹配粒度从字符级子串提升到 token 级，彻底消除误命中。

#### P1-5 异常处理过于宽泛且静默失败 → 已修复

**原始问题**：`except Exception as e` 捕获所有异常；`import logging` 在函数内部；只记录 `warning` 级别；失败后静默继续。

**修复方案**：
- 将 `import logging` / `logger = logging.getLogger(__name__)` 移至文件顶部；
- `except` 改为具体类型 `except (RuntimeError, TypeError, AttributeError)`；
- 改为 `logger.error(..., exc_info=True)` 记录完整堆栈；
- FX trace 失败后 `gm` 保持 `None`，`graph_tensors` 为空，但不再直接返回错误 shape（由 P0-3 修复兜底）。

#### P1-6 违反封装原则：外部直接调用私有方法 → 已修复

**原始问题**：`designer.py` 中 `searcher._extract_op_features(self.meta)` 直接调用以 `_` 开头的私有方法。

**修复方案**：在 `MetaPromptSearcher` 中新增公共方法：
```python
def extract_op_features(self, op_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Public API for extracting operator features from op_meta."""
    return self._extract_op_features(op_meta)
```
`designer.py` 改为调用 `searcher.extract_op_features(self.meta)`，私有实现可独立演进。

### 代码改动

本节记录在上述第一阶段工作（commit `8da1e29a`）基础上，本轮新增的具体代码改动，方便追踪增量变更。

####  `extractor_torch.py`：引入 AST 静态分析，构建双路并行提取

**原有状态**：仅依赖 FX `symbolic_trace` + `ShapeProp` 提取语义；FX trace 失败时 `graph_tensors` 为空，`static_feature_hints` 中所有布尔值全部退化为 `False`，语义信息完全丢失。

**本轮改动**：

1. **新增 `_ast_extract_op_hints(tree)`**：对 KernelBench 算子源码的 AST 进行静态扫描，无需执行代码，识别三类调用模式并直接映射到语义标签：
   - `nn.Xxx`（`__init__` 中的模块实例化）：如 `nn.Conv2d` → `is_conv`，`nn.BatchNorm2d` → `has_online_reduction`
   - `F.xxx`（`forward` 中的 functional 调用）：如 `F.softmax` → `has_online_reduction`
   - `torch.xxx`（顶层函数）：如 `torch.matmul` → `is_matrix`
   - 自动解析 `import torch.nn as nn` / `import torch.nn.functional as F` 等别名，支持 `torch.nn.Xxx` 完整路径

2. **`_validate_code_safety()` 改为返回 `ast.AST`**：原先返回 `None`，现在返回已解析好的 AST 树，供下游 `_ast_extract_op_hints()` 直接复用，避免对同一段代码进行二次 `ast.parse()`。

3. **`_build_static_feature_hints()` 新增 `ast_hints` 参数**：将 FX 结果与 AST 结果取 OR 合并。FX trace 成功时 AST 作为补充信号；FX trace 失败时 AST 成为**唯一**语义来源，保证 `static_feature_hints` 不全为 `False`。

4. **`extract_kernelbench_shapes_dtypes()` 主函数**：调用链改为 `tree = _validate_code_safety(code_str)` → `ast_hints = _ast_extract_op_hints(tree)` → `_build_static_feature_hints(..., ast_hints)`，完成双路整合。

#### `manager.py`：算子语义识别精准化与公共接口暴露

**原有状态**：`_extract_op_features()` 内部使用字符串 `in` 子串匹配做语义识别，存在误命中风险（如 `"mm"` 子串匹配 `"comma"`）；Epilogue 检测无位置约束；`_extract_op_features` 为私有方法，外部调用需直接访问私有接口。

**本轮改动**：

1. **新增 `SEMANTIC_RULES` 类级词表**：将五类语义的关键词（`matrix`、`conv`、`epilogue`、`reduction`、`online_reduction`）集中配置，按 `exact` / `prefix` / `suffix` 三种模式组织。扩展算子识别只需改此词表，无需改动流程逻辑。

2. **新增 `_tokenize_target(target)`**：将算子名拆分为 token 列表，同时展开下划线子 token（如 `conv_transpose2d` → `["conv_transpose2d", "conv", "transpose2d", "2d"]`），使匹配粒度精确到 token 而非子串。

3. **新增 `_matches_semantic_rule(target, exact, prefix, suffix)`**：基于 token 列表的通用语义匹配器，被所有算子类别复用。

4. **新增 `_rule_matches(target, rule_name)`**：按名称查找 `SEMANTIC_RULES` 词表并调用 `_matches_semantic_rule()`；`_extract_op_features()` 中所有语义判断由 `self._rule_matches()` 统一完成，替代原有的散落子串匹配。

5. **Epilogue 检测增加位置感知**：在 `_extract_op_features()` 中，只有出现在至少一个主计算节点（矩阵/卷积）**之后**的逐点算子才被判定为 Epilogue，避免将算子开头的激活函数误报。

6. **新增 `extract_op_features()` 公共方法**：作为 `_extract_op_features()` 的外部访问入口，外部代码（如 `designer.py`）改为调用公共方法，私有实现可独立演进。