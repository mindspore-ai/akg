[English Version](../Architecture.md)

# 框架架构

## 1. 概述

AKG Agents 是一个面向 **AI Infra 与高性能计算**场景的 LLM 多 Agent 协作框架，致力于通过智能 Agent 协同提升高性能代码的开发与优化效率。

框架提供完整的 Agent 基础设施：可扩展的 **Skill / Tools / Sub-agent** 机制、LangGraph 工作流编排、树状 Trace 追踪系统，以及统一的配置与注册体系。

## 2. 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         AKG Agents                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Agents   │  │  Skills   │  │  Tools   │  │   Workflows   │  │
│  │          │  │          │  │          │  │  (LangGraph)  │  │
│  │ AgentBase │  │ Registry │  │ Executor │  │  BaseWorkflow │  │
│  │ ReAct    │  │ Loader   │  │ Basic    │  │  BaseTask     │  │
│  │ Plan     │  │ Selector │  │ Domain   │  │  Router       │  │
│  │ Registry │  │ Hierarchy│  │          │  │  Visualizer   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │              │             │               │            │
│  ┌────┴──────────────┴─────────────┴───────────────┴────────┐  │
│  │                     Trace System                          │  │
│  │  TraceSystem · FileSystemState · ActionCompressor         │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┴────────────────────────────────┐  │
│  │                    LLM Layer                               │  │
│  │  LLMProvider · LLMClient · Embedding                      │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┴────────────────────────────────┐  │
│  │                    配置系统                                 │  │
│  │  AKGSettings · ModelConfig · EmbeddingConfig               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                       场景层                                     │
│  ┌───────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  Kernel Agent (op) │  │  Common Agent  │  │  更多 ...     │  │
│  │  多后端             │  │                │  │              │  │
│  │  多 DSL            │  │                │  │              │  │
│  └───────────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 模块概览

| 模块 | 说明 |
|------|------|
| **Agents** | Agent 基类（`AgentBase`、`ReActAgent`）、Agent 注册与发现机制。详见 [Agent 体系](./AgentSystem.md)。 |
| **Skills** | Skill 管理系统：元数据、加载、注册、层级关系、LLM 驱动选择、版本管理。详见 [Skill 系统](./SkillSystem.md)。 |
| **Tools** | 工具执行框架：内置工具（文件读写、Shell）、领域工具（算子验证、性能分析）、参数解析器。详见 [Tools 体系](./Tools.md)。 |
| **Workflows** | 基于 LangGraph 的工作流编排：`BaseWorkflow`、`BaseLangGraphTask`、路由器、可视化。详见 [工作流](./Workflow.md)。 |
| **Trace** | 树状推理追踪系统：多分叉、状态持久化、断点续跑。详见 [Trace 系统](./Trace.md)。 |
| **LLM** | LLM 接入层：OpenAI 兼容 Provider、带 Token 计数和流式输出的 Client、Embedding 模型。详见 [LLM 接入](./LLM.md)。 |
| **配置系统** | 统一配置管理：`settings.json`、环境变量、多层级优先级。详见 [配置系统](./Configuration.md)。 |

## 4. 场景：Kernel Agent

当前已落地场景为 **AI 算子代码生成** —— 通过 LLM 规划与多 Agent 协同，实现多后端、多 DSL 的高性能算子自动生成与优化。

详见 [Kernel Agent](./KernelAgent.md)。

## 5. CLI

AKG Agents 提供命令行工具（`akg_cli`）用于交互式使用。详见 [AKG CLI](./AKG_CLI.md)。

## 6. 其他模块（v1 文档）

以下模块自 v1 以来未发生变更，文档保留在 v1 目录中：

| 模块 | 说明 | 文档链接 |
|------|------|----------|
| **RAG** | 向量检索增强生成模块 | [RAG (EN)](../v1/RAG.md) / [RAG (CN)](../v1/CN/RAG.md) |
| **Database** | 数据库基类与算子专用存储 | [Database (EN)](../v1/Database.md) / [Database (CN)](../v1/CN/Database.md) |
| **服务化架构** | Client-Server-Worker 架构 | [ServerArchitecture (EN)](../v1/ServerArchitecture.md) / [ServerArchitecture (CN)](../v1/CN/ServerArchitecture.md) |
| **DevicePool** | 设备池管理（Ascend / CUDA / CPU） | [DevicePool (EN)](../v1/DevicePool.md) / [DevicePool (CN)](../v1/CN/DevicePool.md) |
| **TaskPool** | 异步任务池管理 | [TaskPool (EN)](../v1/TaskPool.md) / [TaskPool (CN)](../v1/CN/TaskPool.md) |
