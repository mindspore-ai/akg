# core_v2/agents/ — Agent 开发规范

## 职责

定义 Agent 基类、继承体系和注册机制。

## 继承体系

```
AgentBase                          # base.py — 所有 Agent 的根
├── ReActAgent                     # react_agent.py — ReAct 循环 + 工具调用
├── PlanAgent                      # plan.py — 规划型 Agent
└── SkillEvolutionBase             # skill_evolution_base.py — Skill 进化基类
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `base.py` | `AgentBase`、`Jinja2TemplateWrapper`、`LLMAPIError` |
| `react_agent.py` | `ReActAgent`（继承 AgentBase，内置工具调用循环） |
| `plan.py` | `PlanAgent` |
| `skill_evolution_base.py` | `SkillEvolutionBase` |
| `registry.py` | `AgentRegistry`、`@register_agent` 装饰器 |

## 开发约定

### 新增 Agent 的标准流程

1. 在业务目录（如 `op/agents/`）创建文件，继承 `AgentBase` 或 `ReActAgent`
2. 实现 `run()` 方法
3. 使用 `@register_agent` 注册到 `AgentRegistry`
4. Prompt 模板放在 `op/resources/prompts/<agent_name>/`
5. 通过 `self.load_template()` 加载模板（基于 `get_prompt_path()` 解析路径）

### AgentBase 核心 API

- `load_template(path)` → 从 prompt 根目录加载 Jinja2 模板
- `run()` → Agent 执行入口（子类必须实现）
- LLM 调用通过 `LLMClient` 完成

## 不做什么

- **不要**直接修改 `AgentBase` 的核心行为——通过继承扩展
- **不要**在此目录放业务 Agent（如 KernelGen）——归 `op/agents/`
- **不要**在此目录放测试——归 `tests/ut/`
