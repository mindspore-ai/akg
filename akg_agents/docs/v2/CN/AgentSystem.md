[English Version](../AgentSystem.md)

# Agent 体系

## 1. 概述

Agent 体系提供了 AKG Agents 中构建智能 Agent 的基础设施，包括：

- **AgentBase** — 抽象基类，提供 LLM 调用、模板加载、工具方法等通用能力
- **ReActAgent** — ReAct（Reasoning + Acting）循环实现，用于工具调用型 Agent
- **PlanAgent** — 任务规划 Agent（TODO：最终形态待确定）
- **KernelGen** — 基于 Skill 系统的内核代码生成 Agent（多 DSL、多框架）
- **KernelDesigner** — 基于 Skill 系统的算法草图设计 Agent（硬件感知、Hint 模式）
- **AgentRegistry / register_agent** — Agent 注册与发现机制，支持 scope 隔离

### 导入路径

```python
# 核心类（从顶层包导入）
from akg_agents.core_v2 import AgentBase, Jinja2TemplateWrapper, AgentRegistry, register_agent

# ReActAgent（从 agents 子模块导入 —— 未在顶层导出）
from akg_agents.core_v2.agents import ReActAgent

# PlanAgent
from akg_agents.core_v2.agents import PlanAgent
```

## 2. AgentBase

`AgentBase` 是所有 Agent 的抽象基类，提供每个 Agent 都需要的通用能力。

### 构造函数

```python
class AgentBase(ABC):
    def __init__(self, context: dict = None, config: dict = None):
        """
        Args:
            context: Agent 上下文（agent_name, session_id 等）
            config: 配置信息（agent_model_config, docs_dir 等）
        """
```

### 核心方法

| 方法 | 说明 |
|------|------|
| `run_llm(prompt, input, model_level)` | 调用 LLM，返回 `(content, formatted_prompt, reasoning_content)`。支持模型级别：`"complex"` / `"standard"` / `"fast"`。 |
| `load_template(template_path, template_format)` | 从 prompts 目录加载 Jinja2 模板，返回 `Jinja2TemplateWrapper`。 |
| `load_doc(doc_path)` | 从配置的 docs 目录加载资源文档。 |
| `split_think(content)` | 分离 LLM 输出中的 `</think>` 标签，返回 `(content, reasoning_content)`。 |
| `count_tokens(text, model_name, context)` | 使用 tiktoken 计算 token 数量（未安装时回退到 `len(text) // 4`）。 |
| `load_tool_config()` | 类方法。将 Agent 的工具配置加载为字典（需要定义 `TOOL_NAME`、`DESCRIPTION`、`PARAMETERS_SCHEMA` 类属性）。 |

### Jinja2TemplateWrapper

原生 Jinja2 模板包装器，兼容 LangChain `PromptTemplate` 接口。与 LangChain 的 `SandboxedEnvironment` 不同，它支持完整的 Jinja2 功能（如 `loop.index`）。

```python
from akg_agents.core_v2.agents import Jinja2TemplateWrapper

template = Jinja2TemplateWrapper("Hello {{ name }}, you have {{ items|length }} items.")
result = template.format(name="Alice", items=[1, 2, 3])
```

### 流式输出控制

流式输出通过优先级链控制：

1. **ContextVar 覆盖**（最高优先级）
2. **环境变量** `AKG_AGENTS_STREAM_OUTPUT`
3. **settings.json** 中的 `stream_output` 字段
4. **默认值**：`False`

## 3. ReActAgent

`ReActAgent` 继承 `AgentBase`，实现 ReAct（Reasoning + Acting）模式 —— Agent 推理下一步动作、选择并执行工具、观察结果、循环往复。

### 核心流程（节点优先）

```
┌─────────────────────────────────────────────────┐
│                  ReAct 循环                       │
│                                                  │
│  1. 创建 trace 节点 → 获取 cur_path               │
│  2. 将 cur_path 注入 prompt → 调用 LLM            │
│  3. 解析响应 → 获取 tool_name + arguments          │
│  4. 将 cur_path 注入 arguments → 执行工具          │
│  5. 保存 result.json → 更新 trace 节点            │
│  6. 重复直到完成或达到最大迭代次数                    │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 构造函数

```python
class ReActAgent(AgentBase, ABC):
    def __init__(
        self,
        task_id: str,
        model_level: str = None,
        config: Dict = None,
        base_dir: Optional[str] = None
    ):
```

### 抽象方法（子类必须实现）

| 方法 | 说明 |
|------|------|
| `_get_agent_name()` | 返回 Agent 名称字符串。 |
| `_load_prompt_template()` | 返回系统 prompt 的 `Jinja2TemplateWrapper`。 |
| `_build_prompt_context()` | 返回用于渲染 prompt 模板的变量字典。 |
| `_get_agent_context()` | 返回传递给 `ToolExecutor` 的上下文字典。 |
| `_load_available_tools()` | 返回工具定义列表（OpenAI function calling 格式）。 |

### 可选覆盖方法

| 方法 | 默认值 | 说明 |
|------|--------|------|
| `_load_agent_registry()` | 加载所有已注册 Agent | 加载 Agent 注册表，用于子 Agent 调用。 |
| `_load_workflow_registry()` | 返回 `{}` | 加载 Workflow 注册表。 |
| `_get_task_info_extra()` | 返回 `{}` | 添加额外的任务信息字段。 |
| `_get_domain()` | 返回 `"common"` | 返回领域标识（如 `"op"`、`"graph"`）。 |
| `_on_plan_updated(result)` | 无操作 | Plan 工具执行完成后的回调。 |

### 入口方法

```python
result = await agent.run(user_input="帮我生成一个 relu 算子")
```

`run()` 方法执行完整的 ReAct 循环并返回结果字典。

## 4. PlanAgent

> **TODO**：PlanAgent 的最终形态待确定。当前实现提供基础的任务规划能力。

`PlanAgent` 是一个已注册的 Agent，用于分析用户需求并生成高层次任务规划。它作为工具被主 ReActAgent 调用，不直接与用户交互。

## 5. KernelGen

`KernelGen` 是基于 Skill 系统的内核代码生成 Agent。它继承自 `AgentBase`，通过 Skill 系统动态选择相关知识和策略，生成高性能内核代码。

### 核心能力

- **Skill 驱动的代码生成**：两阶段 Skill 选择（粗筛 + LLM 精筛），动态加载最相关的知识
- **多 DSL 支持**：Triton CUDA、Triton Ascend、AscendC、CUDA C、TileLang CUDA、C++
- **多框架适配**：PyTorch、MindSpore、NumPy
- **工具模式**：注册为 `call_kernel_gen` 工具，可在 KernelAgent 的 ReAct 循环中被调用

### 架构位置

```
AgentBase (core_v2)
    ↑
KernelGen (op/agents/kernel_gen.py)
    ↑
KernelAgent（ReAct Agent，调用 KernelGen 作为 tool）
```

完整详情（参数、Skill 集成、执行流程、使用示例）请参阅 [KernelGen 设计文档](./KernelGen.md)。

## 6. KernelDesigner

`KernelDesigner` 是基于 Skill 系统的算法草图设计 Agent。它继承自 `AgentBase`，根据用户输入和历史上下文，生成算法草图（伪代码 + 优化策略 + 实现建议）。

### 核心能力

- **智能算法草图设计**：根据任务需求生成高质量的算法草图和优化策略
- **Skill 驱动的知识选择**：必选 `sketch-design` Skill + LLM 精筛参考 Skills
- **Hint 模式**：支持参数空间配置，用于引导优化
- **硬件感知设计**：自动加载目标架构的硬件文档
- **工具模式**：注册为 `call_kernel_designer` 工具，可在 KernelAgent 的 ReAct 循环中被调用

### 架构位置

```
AgentBase (core_v2)
    ↑
KernelDesigner (op/agents/kernel_designer.py)
    ↑
KernelAgent（ReAct Agent，调用 KernelDesigner 作为 tool）
```

完整详情（参数、Skill 集成、执行流程、使用示例）请参阅 [KernelDesigner 设计文档](./KernelDesigner.md)。

## 7. AgentRegistry

`AgentRegistry` 是管理 Agent 类的单例注册中心，支持：

- **注册**：通过装饰器或手动调用
- **发现**：列出所有已注册 Agent，可按 scope 过滤
- **创建**：根据名称创建 Agent 实例
- **Scope**：按应用领域隔离 Agent（如 `"op"`、`"common"`）

### 注册方式

```python
from akg_agents.core_v2.agents import AgentBase, register_agent

# 方式 1：无 scope（全局可见）
@register_agent
class MyAgent(AgentBase):
    ...

# 方式 2：指定 scope
@register_agent(scopes=["op"])
class KernelCoder(AgentBase):
    ...

# 方式 3：多个 scope
@register_agent(scopes=["op", "common"])
class SharedAgent(AgentBase):
    ...

# 方式 4：自定义名称 + scope
@register_agent("CustomName", scopes=["op"])
class Coder(AgentBase):
    ...
```

### API 参考

| 方法 | 说明 |
|------|------|
| `AgentRegistry.register(cls, name, scopes)` | 注册 Agent 类。 |
| `AgentRegistry.create_agent(agent_type, **kwargs)` | 根据名称创建 Agent 实例。 |
| `AgentRegistry.list_agents(scope)` | 列出已注册 Agent，可按 scope 过滤。 |
| `AgentRegistry.get_agent_class(name)` | 根据名称获取 Agent 类。 |
| `AgentRegistry.is_registered(name, scope)` | 检查 Agent 是否已注册（在指定 scope 中）。 |
| `AgentRegistry.unregister(name)` | 从注册中心移除 Agent。 |

## 8. 自定义 Agent 示例

### 示例：简单 Agent

```python
from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper

@register_agent(scopes=["my_scenario"])
class MyCustomAgent(AgentBase):
    TOOL_NAME = "call_my_agent"
    DESCRIPTION = "执行自定义任务。"
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "任务描述"
            }
        },
        "required": ["task_description"]
    }

    async def run(self, task_description: str, model_level: str = "standard"):
        template = Jinja2TemplateWrapper("You are a helpful agent.\n\nTask: {{ task }}")
        content, prompt, reasoning = await self.run_llm(
            template,
            {"task": task_description},
            model_level
        )
        return content
```

### 示例：ReAct Agent

```python
from akg_agents.core_v2.agents import ReActAgent, register_agent, Jinja2TemplateWrapper

@register_agent(scopes=["my_scenario"])
class MyReActAgent(ReActAgent):

    def _get_agent_name(self) -> str:
        return "MyReActAgent"

    def _load_prompt_template(self):
        return self.load_template("my_agent/system.j2")

    def _build_prompt_context(self):
        return {"task_id": self.task_id}

    def _get_agent_context(self):
        return {"agent_name": self._get_agent_name()}

    def _load_available_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }
        ]
```
