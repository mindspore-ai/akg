[English Version](../KernelGen.md)

# KernelGen Agent 设计文档

## 概述
KernelGen 是 AKG Agents 中基于 Skill 系统的内核代码生成 Agent。它继承自 `AgentBase`（`core_v2`），负责根据用户输入和历史上下文，利用 Skill 系统动态选择相关知识和策略，生成高性能内核代码。

## 核心功能
- **分层分阶段 Skill 选择**：三层 skill 选择（L0 固定注入、L1 LLM 选择 guide+example、L2 LLM 选择 case），按生成阶段（initial/debug/optimize）自适应
- **Backend 粗筛**：分阶段选择前，先按 backend 元数据预过滤
- **AB Test 支持**：通过 `exclude_skill_names` / `force_skill_names` 精准控制 evolved skill 的 A/B 测试
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
| extra_skills | list (可选) | 额外注入的 skill 对象列表，跳过所有筛选直接追加 |
| exclude_skill_names | list[str] (可选) | 排除指定 skill 名称列表（AB test A 模式） |
| force_skill_names | list[str] (可选) | 强制导入指定 skill 名称列表（AB test B 模式） |

## Skill 系统集成

### 分层分阶段 Skill 选择

KernelGen 使用分层分阶段 Skill 选择机制，按当前生成阶段（`initial`、`debug`、`optimize`）自适应：

**预过滤**：通过 `OperatorSkillSelector.coarse_filter()` 按 backend 元数据移除不兼容的 skill。

**Layer 0（固定注入）**：`fundamental` 和 `reference` 类别的 skill 始终包含，不受阶段影响。

**Layer 1（LLM 选择 guide + example）**：`guide` 类别的 skill 由 LLM 根据任务描述和算子特征选择。`example` 类别的 skill 根据所选 guide 的 `operator_type` 自动匹配。

**Layer 2（LLM 选择 case）**：`case` 类别的 skill 仅在 `debug`（fix case）和 `optimize`（improvement case）阶段包含，与 guide 在同一次 LLM 调用中选择。

**阶段映射**：

| 阶段 | 触发条件 | 包含的类别 |
|------|---------|-----------|
| `initial` | 首次生成 | fundamental, reference, guide, example |
| `debug` | 存在 `verifier_error` | fundamental, reference, guide, example, case (fix) |
| `optimize` | 存在 `inspirations` | fundamental, reference, guide, example, case (improvement) |

**AB Test 控制**：
- `exclude_skill_names`：匹配的 skill 在选择前被移除（A 模式 —— 不含 evolved skill 的基准）
- `force_skill_names`：匹配的 skill 在 LLM 选择后强制追加（B 模式 —— 确保 evolved skill 被包含）

可设置为实例属性或作为 `run()` 参数传入（run 参数临时覆盖实例属性）。

```python
kernel_gen = KernelGen()

# AB test A 模式：排除 evolved skills
kernel_gen.exclude_skill_names = ["triton-ascend-error-fix", "triton-ascend-case-reduce-opt"]

# AB test B 模式：强制导入 evolved skills
kernel_gen.force_skill_names = ["triton-ascend-error-fix"]

# Extra skills：跳过所有筛选，选择后直接追加
kernel_gen.extra_skills = [my_custom_skill]
```

### Skills 目录
Skills 存储在 `op/resources/skills/` 目录下。Evolved skill 可通过软链接从 `~/.akg/evolved_skills/{dsl}/` 链入标准 skill 目录以自动发现。详见 [Skill System 文档](./SkillSystem.md)。

## 执行流程

1. **初始化阶段**
   - 初始化父类 `AgentBase`（配置 LLM、加载模板等）
   - 通过 `parser_loader` 创建代码解析器
   - 加载 Skill 系统（`SkillLoader` 从 skills 目录加载）
   - 加载 Jinja2 Prompt 模板（`system_prompt.j2`、`user_prompt.j2`）

2. **Skill 选择阶段**
   - 先执行 backend 粗筛，再按阶段分层选择（L0 固定注入、L1 LLM 选择 guide+example、L2 LLM 选择 case）
   - 选择前应用 `exclude_skill_names`，选择后应用 `force_skill_names`（AB test 使用）
   - 最后追加 `extra_skills`（如有），确保指定 Skill 一定被选中
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
