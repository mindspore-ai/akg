[中文版](./CN/AgentSystem.md)

# Agent System

## 1. Overview

The Agent System provides the foundational infrastructure for building intelligent agents in AKG Agents. It includes:

- **AgentBase** — Abstract base class with LLM invocation, template loading, and utility methods
- **ReActAgent** — ReAct (Reasoning + Acting) loop implementation for tool-using agents
- **PlanAgent** — Task planning agent (TODO: final design pending)
- **AgentRegistry / register_agent** — Agent registration and discovery mechanism with scope support

### Import Paths

```python
# Core classes (from top-level package)
from akg_agents.core_v2 import AgentBase, Jinja2TemplateWrapper, AgentRegistry, register_agent

# ReActAgent (from agents submodule — not exported at top level)
from akg_agents.core_v2.agents import ReActAgent

# PlanAgent
from akg_agents.core_v2.agents import PlanAgent
```

## 2. AgentBase

`AgentBase` is the abstract base class for all agents. It provides common capabilities that every agent needs.

### Constructor

```python
class AgentBase(ABC):
    def __init__(self, context: dict = None, config: dict = None):
        """
        Args:
            context: Agent context (agent_name, session_id, etc.)
            config: Configuration (agent_model_config, docs_dir, etc.)
        """
```

### Core Methods

| Method | Description |
|--------|-------------|
| `run_llm(prompt, input, model_level)` | Call LLM and return `(content, formatted_prompt, reasoning_content)`. Supports model levels: `"complex"`, `"standard"`, `"fast"`. |
| `load_template(template_path, template_format)` | Load a Jinja2 template from the prompts directory. Returns a `Jinja2TemplateWrapper`. |
| `load_doc(doc_path)` | Load a resource document from the configured docs directory. |
| `split_think(content)` | Split `</think>` tags from LLM output, returning `(content, reasoning_content)`. |
| `count_tokens(text, model_name, context)` | Count tokens using tiktoken (falls back to `len(text) // 4`). |
| `load_tool_config()` | Class method. Load agent's tool configuration as a dict (requires `TOOL_NAME`, `DESCRIPTION`, `PARAMETERS_SCHEMA` class attributes). |

### Jinja2TemplateWrapper

A native Jinja2 template wrapper compatible with the LangChain `PromptTemplate` interface. Unlike LangChain's `SandboxedEnvironment`, it supports full Jinja2 features including `loop.index`.

```python
from akg_agents.core_v2.agents import Jinja2TemplateWrapper

template = Jinja2TemplateWrapper("Hello {{ name }}, you have {{ items|length }} items.")
result = template.format(name="Alice", items=[1, 2, 3])
```

### Stream Output Control

Stream output is controlled through a priority chain:

1. **ContextVar override** (highest priority)
2. **Environment variable** `AKG_AGENTS_STREAM_OUTPUT`
3. **settings.json** `stream_output` field
4. **Default**: `False`

## 3. ReActAgent

`ReActAgent` extends `AgentBase` to implement the ReAct (Reasoning + Acting) pattern — a loop where the agent reasons about the next step, selects and executes a tool, observes the result, and repeats.

### Core Flow (Node-First)

```
┌─────────────────────────────────────────────────┐
│                  ReAct Loop                      │
│                                                  │
│  1. Create trace node → get cur_path             │
│  2. Inject cur_path into prompt → call LLM       │
│  3. Parse response → get tool_name + arguments   │
│  4. Inject cur_path into arguments → execute tool │
│  5. Save result.json → update trace node         │
│  6. Repeat until done or max iterations          │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Constructor

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

### Abstract Methods (Must Implement)

| Method | Description |
|--------|-------------|
| `_get_agent_name()` | Return the agent name string. |
| `_load_prompt_template()` | Return a `Jinja2TemplateWrapper` for the system prompt. |
| `_build_prompt_context()` | Return a dict of template variables for prompt rendering. |
| `_get_agent_context()` | Return context dict passed to `ToolExecutor`. |
| `_load_available_tools()` | Return a list of tool definitions (OpenAI function calling format). |

### Optional Override Methods

| Method | Default | Description |
|--------|---------|-------------|
| `_load_agent_registry()` | Loads all registered agents | Load agent registry for sub-agent calls. |
| `_load_workflow_registry()` | Returns `{}` | Load workflow registry. |
| `_get_task_info_extra()` | Returns `{}` | Add extra fields to task info. |
| `_get_domain()` | Returns `"common"` | Return domain identifier (e.g., `"op"`, `"graph"`). |
| `_on_plan_updated(result)` | No-op | Callback when plan tool execution completes. |

### Entry Point

```python
result = await agent.run(user_input="Generate a relu kernel")
```

The `run()` method executes the full ReAct loop and returns a result dict.

## 4. PlanAgent

> **TODO**: The PlanAgent's final design is pending. Current implementation provides basic task planning capability.

`PlanAgent` is a registered agent that analyzes user requirements and generates high-level task plans. It is called as a tool by the main ReActAgent, not directly by users.

## 5. AgentRegistry

`AgentRegistry` is a singleton registry for managing agent classes. It supports:

- **Registration**: via decorator or manual call
- **Discovery**: list all registered agents, optionally filtered by scope
- **Creation**: instantiate agents by name
- **Scope**: isolate agents by application domain (e.g., `"op"`, `"common"`)

### Registration

```python
from akg_agents.core_v2.agents import AgentBase, register_agent

# Method 1: No scope (globally visible)
@register_agent
class MyAgent(AgentBase):
    ...

# Method 2: With scope
@register_agent(scopes=["op"])
class KernelCoder(AgentBase):
    ...

# Method 3: Multiple scopes
@register_agent(scopes=["op", "common"])
class SharedAgent(AgentBase):
    ...

# Method 4: Custom name + scope
@register_agent("CustomName", scopes=["op"])
class Coder(AgentBase):
    ...
```

### API Reference

| Method | Description |
|--------|-------------|
| `AgentRegistry.register(cls, name, scopes)` | Register an agent class. |
| `AgentRegistry.create_agent(agent_type, **kwargs)` | Create an agent instance by name. |
| `AgentRegistry.list_agents(scope)` | List registered agents, optionally filtered by scope. |
| `AgentRegistry.get_agent_class(name)` | Get agent class by name. |
| `AgentRegistry.is_registered(name, scope)` | Check if an agent is registered (in a given scope). |
| `AgentRegistry.unregister(name)` | Remove an agent from the registry. |

## 6. Creating a Custom Agent

### Example: Simple Agent

```python
from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper

@register_agent(scopes=["my_scenario"])
class MyCustomAgent(AgentBase):
    TOOL_NAME = "call_my_agent"
    DESCRIPTION = "Performs a custom task."
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Description of the task"
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

### Example: ReAct Agent

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
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }
        ]
```
