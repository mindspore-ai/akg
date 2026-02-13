[中文版](./CN/Architecture.md)

# Architecture

## 1. Overview

AKG Agents is an LLM-powered multi-agent collaboration framework for **AI Infra and high-performance computing**, aimed at boosting the development and optimization efficiency of high-performance code through intelligent agent collaboration.

The framework provides a complete agent infrastructure: extensible **Skill / Tools / Sub-agent** mechanisms, LangGraph workflow orchestration, tree-based Trace system, and a unified configuration and registry.

## 2. Architecture Diagram

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
│  │                  Configuration                             │  │
│  │  AKGSettings · ModelConfig · EmbeddingConfig               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     Scenarios                                   │
│  ┌───────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  Kernel Agent (op) │  │  Common Agent  │  │  More ...     │  │
│  │  Multi-backend     │  │                │  │              │  │
│  │  Multi-DSL         │  │                │  │              │  │
│  └───────────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Module Overview

| Module | Description |
|--------|-------------|
| **Agents** | Agent base classes (`AgentBase`, `ReActAgent`), agent registry and discovery mechanism. See [Agent System](./AgentSystem.md). |
| **Skills** | Skill management system: metadata, loading, registry, hierarchy, LLM-driven selection, version management. See [Skill System](./SkillSystem.md). |
| **Tools** | Tool execution framework: built-in tools (file I/O, shell), domain tools (kernel verification, profiling), argument resolver. See [Tools](./Tools.md). |
| **Workflows** | LangGraph-based workflow orchestration: `BaseWorkflow`, `BaseLangGraphTask`, routers, visualization. See [Workflow](./Workflow.md). |
| **Trace** | Tree-based inference tracing system: multi-fork, state persistence, checkpoint resume. See [Trace System](./Trace.md). |
| **LLM** | LLM access layer: OpenAI-compatible provider, client with token counting and streaming, embedding models. See [LLM](./LLM.md). |
| **Configuration** | Unified configuration management: `settings.json`, environment variables, multi-level priority. See [Configuration](./Configuration.md). |

## 4. Scenario: Kernel Agent

The first production scenario is **AI Kernel Code Generation** — leveraging LLM planning and multi-agent collaboration to automate multi-backend, multi-DSL high-performance kernel generation and optimization.

For details, see [Kernel Agent](./KernelAgent.md).

## 5. CLI

AKG Agents provides a command-line interface (`akg_cli`) for interactive use. See [AKG CLI](./AKG_CLI.md).

## 6. Additional Modules (v1 Documentation)

The following modules have not changed since v1 and their documentation remains in the v1 directory:

| Module | Description | Documentation |
|--------|-------------|---------------|
| **RAG** | Vector retrieval-augmented generation module | [RAG (EN)](../v1/RAG.md) / [RAG (CN)](../v1/CN/RAG.md) |
| **Database** | Database base class and kernel-specific storage | [Database (EN)](../v1/Database.md) / [Database (CN)](../v1/CN/Database.md) |
| **Server Architecture** | Client-Server-Worker service architecture | [ServerArchitecture (EN)](../v1/ServerArchitecture.md) / [ServerArchitecture (CN)](../v1/CN/ServerArchitecture.md) |
| **DevicePool** | Device pool management (Ascend / CUDA / CPU) | [DevicePool (EN)](../v1/DevicePool.md) / [DevicePool (CN)](../v1/CN/DevicePool.md) |
| **TaskPool** | Async task pool management | [TaskPool (EN)](../v1/TaskPool.md) / [TaskPool (CN)](../v1/CN/TaskPool.md) |
