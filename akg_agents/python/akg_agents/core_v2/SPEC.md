# core_v2/ — v2 核心框架

## 职责

提供 Agent、Workflow、Skill、Tool、LLM、Config 等通用基础设施。所有业务场景层（当前：`op/` 算子生成；规划中：图优化等）和 CLI 均构建在此之上。新增业务场景层时应只依赖 `core_v2/` 提供的基类和注册机制，不应依赖 `op/` 中的实现。

## 目录结构

| 子目录 | 职责 | 规范 |
|--------|------|------|
| `agents/` | Agent 基类与注册（AgentBase、ReActAgent、PlanAgent） | [SPEC.md](agents/SPEC.md) |
| `langgraph_base/` | LangGraph 工作流基座（BaseState、BaseWorkflow、BaseLangGraphTask） | — |
| `llm/` | LLM 客户端、Provider 工厂、Embedding | — |
| `skill/` | Skill 注册/加载/层级/安装/选择/版本 | [SPEC.md](skill/SPEC.md) |
| `tools/` | ToolRegistry、ToolExecutor、内置工具 | — |
| `config/` | AKGSettings、配置加载与合并 | — |
| `filesystem/` | Trace、快照/压缩、状态可视化 | — |
| `workflows/` | 通用工作流注册 | — |
| `tests/` | 旧遗留测试（**新测试请放 `tests/ut/`**） | — |

## 开发约定

### 关键基类

- `AgentBase` → 所有 Agent 的根基类
- `BaseWorkflow` → 所有 LangGraph 工作流的基类
- `BaseLangGraphTask` → 任务执行入口的基类
- `ToolExecutor` → 工具执行器（无 BaseTool 抽象类，以 `ToolRegistry` + `ToolInfo` 为中心）

### 配置优先级

环境变量 > Local(`*.local.json`) > Project(`.akg/settings.json`) > User(`~/.akg/settings.json`) > 默认值

### 扩展此框架

1. 新增 Agent → 见 [agents/SPEC.md](agents/SPEC.md)
2. 新增 Skill → 见 [skill/SPEC.md](skill/SPEC.md)
3. 新增 Workflow → 继承 `BaseWorkflow`，实现 `build_graph()`
4. 新增 Tool → 创建 `ToolInfo`，注册到 `ToolRegistry`

## 不做什么

- **不要**在此写业务场景逻辑（算子生成归 `op/`，后续图优化等归各自场景层）
- **不要**在此写 CLI 命令——归 `cli/`
- **不要**在 `tests/` 子目录写新测试——归 `../../tests/`

## 参考

- `docs/v2/Architecture.md` — 整体架构设计
- `docs/v2/AgentSystem.md` — Agent 继承体系
- `docs/v2/Workflow.md` — 工作流设计
