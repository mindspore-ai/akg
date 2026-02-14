# KernelGen Agent 设计文档

## 概述
KernelGen 是 AKG Agents 中基于 Skill 系统的内核代码生成 Agent。它继承自 `AgentBase`（`core_v2`），负责根据用户输入和历史上下文，利用 Skill 系统动态选择相关知识和策略，生成高性能内核代码。

## 核心功能
- **Skill 驱动的代码生成**：通过两阶段 Skill 选择（粗筛 + LLM 精筛），动态加载最相关的知识
- **多 DSL 支持**：支持 Triton CUDA、Triton Ascend、AscendC 等多种 DSL
- **多框架适配**：支持 PyTorch、MindSpore、NumPy 等前端框架
- **Agent 注册机制**：通过 `@register_agent` 装饰器自动注册到 Agent 注册表
- **工具化配置**：提供 `TOOL_NAME`、`DESCRIPTION`、`PARAMETERS_SCHEMA`，可作为 Tool 被 KernelAgent 调用

## 架构位置

```
core_v2/agents/base.py          # AgentBase 基类
    ↑
op/agents/kernel_gen.py          # KernelGen Agent（继承 AgentBase）
    ↑
op/agents/kernel_agent.py        # KernelAgent（ReAct Agent，调用 KernelGen 作为 tool）
```

## 工具配置

KernelGen 作为 Tool 供 KernelAgent 调用时的配置：

| 属性 | 值 |
|------|-----|
| TOOL_NAME | `call_kernel_gen` |
| 适用场景 | 用户明确说"不用验证"、"只给我代码"、"快速生成草稿" |
| 输出 | 生成的 kernel 代码（包含 class ModelNew 和 kernel 函数） |

⚠️ 此工具仅生成代码，不验证正确性和性能。如需完整的生成+验证流程，请使用 `use_kernelgen_only_workflow`。

## 参数说明

| 参数名称 | 类型/必选 | 描述 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称 |
| task_desc | str (必选) | 任务描述或算法规格说明 |
| dsl | str (必选) | 目标 DSL：`triton_ascend`、`triton_cuda` 等 |
| framework | str (必选) | 目标框架：`torch`、`mindspore`、`numpy` 等 |
| backend | str (必选) | 目标硬件后端：`cuda`、`ascend` 等 |
| arch | str (可选) | 目标硬件架构：`a100`、`ascend910b4` 等 |
| user_requirements | str (可选) | 用户额外需求 |
| task_id | str (可选) | 任务 ID |
| history_compress | list (可选) | 压缩后的历史记录列表 |
| verifier_error | str (可选) | Verifier 错误信息 |
| conductor_suggestion | str (可选) | Conductor 修复建议 |
| model_level | str (可选) | 模型级别：`standard`、`fast`、`complex`，默认 `standard` |

## Skill 系统集成

### 两阶段 Skill 选择

KernelGen 使用两阶段 Skill 选择机制：

1. **粗筛（Metadata 过滤）**：使用 `OperatorSkillSelector` 基于 backend / dsl 等 metadata 快速过滤
2. **精筛（LLM 评估）**：如果候选 Skill 超过 3 个，由 LLM 根据任务描述选择最相关的 Skills

```python
# 粗筛
context = OperatorSelectionContext(dsl="triton-ascend", backend="ascend")
filtered = selector.coarse_filter(loaded_skills, context)

# 精筛（候选 > 3 时）
selected = llm_select(filtered, task_desc, op_name)
```

### Skills 目录
Skills 存储在 `op/resources/skills/` 目录下，详见 [Skill System 文档](./SkillSystem.md)。

## 执行流程

1. **初始化阶段**
   - 初始化父类 `AgentBase`（配置 LLM、加载模板等）
   - 通过 `parser_loader` 创建代码解析器
   - 加载 Skill 系统（`SkillLoader` 从 skills 目录加载）
   - 加载 Jinja2 Prompt 模板（`system_prompt.j2`、`user_prompt.j2`）

2. **Skill 选择阶段**
   - 基于任务参数（op_name, dsl, backend 等）执行两阶段 Skill 选择
   - 返回最相关的 Skills 列表

3. **Prompt 构建阶段**
   - 使用 System Prompt 模板渲染系统提示（含 DSL、框架、后端信息）
   - 使用 User Prompt 模板渲染用户提示（含历史记录、Skills 内容、任务描述等）

4. **代码生成阶段**
   - 通过 `run_llm` 调用 LLM 生成代码
   - 返回 `(生成的代码, 完整 prompt, 推理过程)`

## 使用示例

### 直接调用
```python
from akg_agents.op.agents.kernel_gen import KernelGen

# 初始化
kernel_gen = KernelGen()

# 执行代码生成
code, prompt, reasoning = await kernel_gen.run(
    op_name="relu",
    task_desc="实现 ReLU 激活函数",
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100"
)
```

### 通过 KernelAgent 调用（作为 Tool）
KernelGen 被注册为 `call_kernel_gen` 工具，由 KernelAgent 在 ReAct 循环中根据用户需求自动调用。详见 [Workflow 文档](./Workflow.md)。

## 相关文档
- [KernelDesigner 设计文档](./KernelDesigner.md)
- [Workflow 与任务系统](./Workflow.md)
- [Skill System 文档](./SkillSystem.md)
- [Trace System 文档](./Trace.md)
