# op/agents/ — 算子 Agent

## 职责

算子/内核生成场景下的各专业 Agent。所有 Agent 继承 `core_v2` 中的基类。

## Agent 清单

| Agent | 基类 | 职责 |
|-------|------|------|
| `KernelGen` | `AgentBase` | 内核代码生成（写代码） |
| `KernelDesigner` | `AgentBase` | 内核方案设计（出设计方案） |
| `KernelAgent` | `ReActAgent` | ReAct 模式内核 Agent（工具调用） |
| `OpTaskBuilder` | `AgentBase` | 算子任务构建 |
| `TaskConstructor` | `AgentBase` | 任务描述构建（含 `StepRecord`、`SessionLogger`） |
| `SkillEvolutionAgent` | `SkillEvolutionBase` | Skill 自进化 |

## 开发约定

### 新增算子 Agent 的标准流程

1. 在本目录创建 `<agent_name>.py`
2. 继承 `AgentBase`（通用）或 `ReActAgent`（需要工具调用）
3. 使用 `@register_agent` 注册
4. Prompt 模板放在 `../resources/prompts/<agent_name>/`
5. 如果需要加入工作流，在对应 workflow 中添加为节点

## 不做什么

- **不要**修改 Agent 基类（`AgentBase`/`ReActAgent`）——归 `core_v2/agents/`
- **不要**在此写验证逻辑——归 `../verifier/`
- **不要**在此写工作流编排——归 `../workflows/`
