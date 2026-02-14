[English Version](../KernelDesigner.md)

# KernelDesigner Agent 设计文档

## 概述
KernelDesigner 是 AKG Agents 中基于 Skill 系统的算法草图设计 Agent。它继承自 `AgentBase`（`core_v2`），负责根据用户输入和历史上下文，生成算法草图（sketch），提供算法实现方案和优化策略。

## 核心功能
- **智能算法草图设计**：根据任务需求生成高质量的算法草图（伪代码 + 优化策略 + 实现建议）
- **Skill 驱动的知识选择**：自动选择 sketch-design 必选 Skill，并通过 LLM 精筛选择参考 Skills
- **Hint 模式**：支持参数空间配置（Parameter Space），当任务描述中包含 hint 信息时自动启用
- **多 DSL 支持**：支持 Triton CUDA、Triton Ascend 等多种 DSL
- **硬件感知设计**：自动加载硬件文档，在设计过程中考虑硬件特性
- **Agent 注册机制**：通过 `@register_agent` 装饰器自动注册到 Agent 注册表

## 架构位置

```
core_v2/agents/base.py            # AgentBase 基类
    ↑
op/agents/kernel_designer.py       # KernelDesigner Agent（继承 AgentBase）
    ↑
op/agents/kernel_agent.py          # KernelAgent（ReAct Agent，调用 KernelDesigner 作为 tool）
```

## 工具配置

KernelDesigner 作为 Tool 供 KernelAgent 调用时的配置：

| 属性 | 值 |
|------|-----|
| TOOL_NAME | `call_kernel_designer` |
| 适用场景 | 用户说"设计方案"、"算法草图"、"sketch"、"怎么实现"、需要先讨论设计方案 |
| 输出 | 算法草图（伪代码 + 优化策略 + 实现建议） |

⚠️ 此工具仅生成设计方案，不生成可执行代码。如需设计+代码生成，请使用 `use_default_workflow`。

## 参数说明

| 参数名称 | 类型/必选 | 描述 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称 |
| task_desc | str (必选) | 任务描述或算法规格说明 |
| dsl | str (必选) | 目标 DSL：`triton_ascend`、`triton_cuda` 等 |
| backend | str (必选) | 目标硬件后端：`cuda`、`ascend` 等 |
| arch | str (可选) | 目标硬件架构：`a100`、`ascend910b4` 等 |
| task_id | str (可选) | 任务 ID |
| user_requirements | str (可选) | 用户额外需求 |
| enable_hint_mode | bool (可选) | 是否启用 Hint 模式（参数空间配置），默认 False |
| history_compress | list (可选) | 压缩后的历史记录列表 |
| model_level | str (可选) | 模型级别：`standard`、`fast`、`complex`，默认 `standard` |

## Skill 系统集成

### Designer 的 Skill 选择策略

KernelDesigner 使用与 KernelGen 不同的 Skill 选择策略：

1. **必选 Skill**：`sketch-design`（始终加载）
2. **条件 Skill**：`hint-mode`（当 `enable_hint_mode=True` 且检测到 hint 信息时加载）
3. **LLM 精筛**：从粗筛候选中，由 LLM 选择最相关的 2-3 个参考 Skills

```python
# Designer 自定义过滤器
def designer_filter(skill, context):
    # 优先选择 design / fundamental 类别
    # 排除纯 implementation 类别（由 KernelGen 使用）
    design_categories = ["design", "fundamental"]
    is_designer_skill = skill.category in design_categories
    implementation_only = skill.category == "implementation" and not is_designer_skill
    return not implementation_only
```

### 硬件文档加载

KernelDesigner 会根据 backend 和 arch 参数自动加载对应的硬件文档：

```python
from akg_agents.utils.hardware_utils import get_hardware_doc
hardware_docs = get_hardware_doc(backend, arch)
```

## 执行流程

1. **初始化阶段**
   - 初始化父类 `AgentBase`
   - 通过 `parser_loader` 创建 designer 解析器
   - 加载 Skill 系统（带自定义 designer 过滤器）
   - 加载 Jinja2 Prompt 模板（`system_prompt.j2`、`user_prompt.j2`）

2. **Skill 选择阶段**
   - 加载必选 Skill（`sketch-design`）
   - 条件加载 `hint-mode` Skill
   - 使用自定义过滤器粗筛
   - LLM 从候选中精筛 2-3 个参考 Skills

3. **Prompt 构建阶段**
   - 渲染 System Prompt（含 DSL、后端、架构信息）
   - 获取硬件文档
   - 渲染 User Prompt（含历史记录、Skills、任务描述、硬件文档等）

4. **设计生成阶段**
   - 通过 `run_llm` 调用 LLM 生成算法草图
   - 返回 `(生成的草图, 完整 prompt, 推理过程)`

## 使用示例

### 直接调用
```python
from akg_agents.op.agents.kernel_designer import KernelDesigner

# 初始化
designer = KernelDesigner()

# 执行设计
sketch, prompt, reasoning = await designer.run(
    op_name="softmax",
    task_desc="实现 fused softmax 算子",
    dsl="triton_cuda",
    backend="cuda",
    arch="a100"
)
```

### 通过 KernelAgent 调用（作为 Tool）
KernelDesigner 被注册为 `call_kernel_designer` 工具，由 KernelAgent 在 ReAct 循环中根据用户需求自动调用。详见 [Workflow 文档](./Workflow.md)。

## 相关文档
- [KernelGen Agent 设计文档](./KernelGen.md)
- [Workflow 与任务系统](./Workflow.md)
- [Skill System 文档](./SkillSystem.md)
- [Trace System 文档](./Trace.md)
