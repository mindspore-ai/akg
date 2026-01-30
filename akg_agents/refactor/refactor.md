# AIKG 重构架构框架（最终版）

> 基于专家文档（1.md, ARCHITECTURE_CN.md, POC*）的整合方案
> 讨论时间: 2026-01-20

## 核心设计决策

### 1. 编排粒度（混合模式）
```
KernelAgent (手动主循环)
    ↓ 可以调用（都是 tools）
    ├─ Workflow (LangGraph，固化流程) → 简单标准场景
    ├─ SubAgent (LangGraph，单步) → 复杂场景 + HITL
    ├─ Tool (基础工具: read_file, write_file, ...)
    └─ ask_user (HITL 工具)

理念: 
• Workflow = 快捷方式（简单标准场景，一次性走完）
• SubAgent = 细粒度控制（复杂场景 + HITL，逐步调用）
• LLM 自主决策用哪个

关键特性:
• KernelAgent 不基于 LangGraph（手动主循环）
• 使用 OpenAI function calling (ReAct 风格)
• PlanAgent + 每10步固化（定期规划机制）
• FileSystem 作为状态存储
     
注: 不同任务场景有不同 Agent (如 KernelAgent, DocAgent...)
```

### 2. 文件系统
```
~/.akg/conversations/{task_id}/
├─ trace.json              # Trace树(状态+调用关系)
├─ state.json              # 当前状态快照
├─ thinking.json           # 固化的关键信息(每10步)
├─ actions/                # 动作历史
├─ code/                   # 生成的代码
└─ logs/                   # 日志

核心理念: 文件系统状态 = 所有历史操作的效果
```

### 3. Trace 统一概念
```
Trace = 状态树 + 调用树 (合二为一)
├─ 每个节点: 一个状态快照 + Agent/Workflow 调用信息
├─ 支持多方案探索: 树的不同分支
└─ 支持回溯: 返回任意历史节点
```

### 4. 迁移策略
```
阶段1: 算子专用内容迁移到 ops/
阶段2: 通用框架重构 core_v2/
阶段3: 通用场景层建设 common/ (NEW)
同时进行: 全面 Skills化/Tools化/Agents化

目标: 逐步迁移,新旧目录并存过渡
- core_v2/ (新通用框架 - 不依赖任何领域)
- ops/     (算子场景 - akg_cli op)
- common/  (通用场景 - akg_cli common，文档、重构、测试等)
- 保留原有代码结构,逐步废弃

关键设计:
• ops/ 和 common/ 是同级目录，都基于 core_v2/
• akg_cli op 和 akg_cli common 是同级命令
• 通过 common/ 验证 core_v2 的通用框架能力
```

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    【CLI 统一入口 - 全新交互】                     │
│                                                                 │
│  akg_cli op [启动算子场景]                                       │
│    ├─ 斜杠命令交互: /trace, /parallel, /help, /mode...          │
│    ├─ 自动补全: Tab 补全命令和参数                               │
│    ├─ 命令历史: 上下箭头浏览历史                                  │
│    └─ 智能面板: F2 切换显示/隐藏                                  │
│                                                                 │
│  akg_cli common [启动通用场景] (NEW)                             │
│    ├─ 相同的斜杠命令交互体验                                      │
│    ├─ 支持非算子场景                                             │
│    ├─ 动态加载外部 Skills                                        │
│    └─ Skills 组合使用                                            │
│                                                                 │
│  设计理念: 去掉 Top5 框架，统一斜杠命令交互                        │
│  详见: CLI_DESIGN.md                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    【HITL 编排层】(NEW)                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ PlanAgent    │  │ KernelAgent  │  │ Trace System │          │
│  │ 每10步规划   │  │ 主循环+决策  │  │ 多方案探索   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  KernelAgent (手动主循环 + OpenAI function calling):            │
│    • Workflow tools (简单场景: use_standard_workflow)          │
│    • SubAgent tools (复杂场景: call_designer, call_coder...)  │
│    • 基础 tools (read_file, write_file, ask_user...)          │
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │ FileSystemState (~/.akg/conversations/)  │                  │
│  │ • trace.json (Trace树)                   │                  │
│  │ • thinking.json (PlanAgent固化)          │                  │
│  │ • actions/ (每个action详细记录)          │                  │
│  │ • code/ (生成的代码)                      │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │ Skills System (动态加载、SubSkills组合)   │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  【通用框架层 - core_v2/】                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Agent 框架                                       │           │
│  │  • AgentBase (100行,精简版)                     │           │
│  │  • AgentRegistry (@register_agent)             │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Workflows (LangGraph StateGraph)               │           │
│  │  • 保持现有 BaseWorkflow 不变                  │           │
│  │  • 作为 tools 被 KernelAgent 调用              │           │
│  │  • 适用于简单标准场景                           │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ LLM 抽象层                                       │           │
│  │  • LLMProvider (统一接口)                       │           │
│  │  • LLMClient (Token管理、流式)                  │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ 状态管理                                         │           │
│  │  • Trace System (状态树+多方案探索)             │           │
│  │  • StateStorage (状态持久化)                    │           │
│  │  • PlanAgent (规划+每10步固化)                  │           │
│  │  • ActionCompressor (智能压缩)                  │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Tools 系统                                       │           │
│  │  • Base Tools (通用工具)                        │           │
│  │    - FileTools (文件操作)                       │           │
│  │    - CompilerTools (编译执行)                   │           │
│  │  • Domain Tools (领域专用,动态加载)              │           │
│  │    - verifyTool (验证工具)                      │           │
│  │    - 其他专用工具                                │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Worker 系统 (保持现有)                           │           │
│  │  • WorkerManager (本地/远程)                    │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 【算子专用层 - ops/】                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ 算子 Agents (SYSTEM_PROMPT + Skills 组合)        │           │
│  │  • KernelAgent (主Agent)                        │           │
│  │  • Designer SubAgent                            │           │
│  │  • Coder SubAgent                               │           │
│  │  • 其他专用 SubAgent                             │           │
│  │  (Skills 动态加载,不同任务不同组合)               │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ 算子 Tools                                       │           │
│  │  • verifyTool (Kernel验证)                      │           │
│  │  • compileTool (编译工具)                       │           │
│  │  • profileTool (性能分析)                       │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ 固定工作流                                        │           │
│  │  • StandardOpWorkflow (标准算子生成)             │           │
│  │  • EvolveWorkflow (优化迭代)                     │           │
│  │  • SearchWorkflow (方案搜索)                     │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Kernel Skills 系统                               │           │
│  │  • 编译器知识 (CUDA/Ascend/...)                  │           │
│  │  • DSL 知识 (Triton/SWFT/...)                   │           │
│  │  • 优化模式库                                     │           │
│  │  • 示例代码库                                     │           │
│  │  • 最佳实践文档                                   │           │
│  │  • 动态加载,按需组合                              │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                            ↓ (同级关系)
┌─────────────────────────────────────────────────────────────────┐
│              【通用场景层 - common/】(NEW)                        │
│              与 ops/ 同级，都基于 core_v2/                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ 通用 Agents                                      │           │
│  │  • 基于 core_v2/ 框架                           │           │
│  │  • 支持外部 Skills 动态加载                      │           │
│  │  • 各类通用场景的 Agent 实现                     │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  核心价值:                                                       │
│  • 验证 core_v2 的通用性（不绑定算子场景）                       │
│  • 支持加载和使用社区 Skills                                     │
│  • 快速扩展到非算子场景                                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    【文件系统层】                                 │
│                                                                 │
│  ~/.akg/conversations/{task_id}/                              │
│    ├─ trace.json         (Trace树: 状态+调用)                  │
│    ├─ state.json         (当前状态快照)                        │
│    ├─ thinking.json      (每10步固化)                          │
│    ├─ actions/           (动作历史)                            │
│    ├─ code/              (生成的代码)                          │
│    └─ logs/              (日志)                                │
│                                                                 │
│  理念: 文件系统 = 真相源, Agent 通过读文件了解历史               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五大重构任务

### 任务1: 核心框架解耦 (core_v2/)
**目标**: 通用框架,可用于非算子场景

**产出**:
- AgentBase (100行精简版)
- AgentRegistry (装饰器注册)
- LLM 抽象层 (统一接口)
- Orchestrator (对 StateGraph 封装,提供 API)
- WorkflowBuilder (原 BaseWorkflow,改名)
- WorkerManager (保持现有,不改)
- Base Tools (通用基础工具)
- Domain Tools 加载机制

**关键**: 不依赖任何算子专用逻辑,充分复用现有代码

---

### 任务2: HITL 编排层建设
**目标**: 增强人机协作和智能决策

**产出**:
- PlanAgent (规划+每10步固化)
- KernelAgent (算子场景主Agent,基于LangGraph ReAct)
- Trace System (状态树+多方案探索)
- ActionCompressor (智能压缩)

**关键**: 支持 Tools/SubAgents/Orchestrator 三种调用方式

---

### 任务3: 文件系统状态管理
**目标**: 实现无限运行能力

**产出**:
- `~/.akg/conversations/` 文件结构
- Trace System (状态树+多方案管理)
- StateStorage (状态持久化)
- 断点续跑机制

**关键**: 文件系统 = 所有历史操作的效果,结合LangGraph Checkpointer

---

### 任务4: Skills 系统
**目标**: 知识模块化和动态组合

**产出**:
- Skills 动态加载机制
- SubSkills 组合能力
- 所有现有文档 Skills 化

**关键**: 减少 token 消耗,按需加载

---

### 任务5: 算子层全面改造 (ops/)
**目标**: 将算子能力迁移到新架构

**产出**:
- KernelAgent (SYSTEM_PROMPT + Skills 动态组合)
- SubAgents (Designer, Coder等)
- verifyTool / compileTool / profileTool (Tools化)
- Kernel Skills 系统 (编译器、DSL、优化模式等)
- Orchestrator 固化标准流程

**策略**: 
1. 验证逻辑 Tools 化
2. Agent 从 Skills 动态组合
3. 编译器/DSL 归入 Skills

---

### 任务6: 通用场景层建设 (common/) (NEW)
**目标**: 验证框架通用性，支持非算子场景

**说明**: common/ 与 ops/ 同级，都基于 core_v2/，对应 `akg_cli common` 命令

**产出**:
- 通用场景的 Agents 实现
- 外部 Skills 加载接口
- Skills 动态组合能力
- `akg_cli common` 命令支持

**核心价值**:
- ✅ 证明 core_v2 的通用性（不绑定算子场景）
- ✅ 支持加载和使用社区 Skills
- ✅ 快速扩展到非算子场景

**注**: 具体 Skills 格式和加载逻辑由实现者设计

---

### 任务7: CLI 交互层重构 (cli/) (NEW)
**目标**: 统一斜杠命令交互，去掉 Top5 框架，提升用户体验

**说明**: 参考 qwen-agent 的命令式交互风格，基于现有 prompt_toolkit 架构轻量改造

**产出**:
- 斜杠命令系统 (`/trace`, `/parallel`, `/help`, `/mode` 等)
- 命令自动补全器（Tab 补全 + 描述提示）
- 命令注册机制（装饰器风格，易扩展）
- 简化面板显示（移除 Top5 框架）
- 统一交互体验（op 和 common 命令相同）

**核心价值**:
- ✅ 更直观的命令式交互（斜杠命令 vs 自然语言混合）
- ✅ 强大的自动补全（提升效率）
- ✅ 易扩展（新命令通过装饰器注册）
- ✅ 统一体验（op 和 common 一致）

**技术栈**: prompt_toolkit 3.0 + typer + rich（无需新增依赖）

**详细设计**: 参见 `CLI_DESIGN.md`

---

## 关键机制

### 1. 混合调用模式
```
KernelAgent 主循环 (手动实现):
  for turn in range(max_turns):
      # 1. 主动检查 plan 是否过期（规则检测）
      if is_plan_outdated(plan, action_history):
          plan = plan_agent.replan(...)
          action_history = []
      
      # 2. 构建上下文 (System Prompt)
      context = build_context(plan, action_history, file_state)
      
      # 3. LLM 决策 (OpenAI function calling)
      response = llm.chat(system_prompt=context, tools=all_tools)
      
      # 4. 执行 tools
      ask_user_called = False
      for tool_call in response.tool_calls:
          if tool_name.startswith("use_") and tool_name.endswith("_workflow"):
              # 调用 Workflow (简单标准场景)
              result = workflow.ainvoke(...)
          
          elif tool_name.startswith("call_"):
              # 调用 SubAgent (复杂场景 + HITL)
              result = subagent.execute(...)
          
          elif tool_name == "ask_user":
              # HITL (暂停执行，等待用户输入)
              result = await wait_for_user_input(...)
              ask_user_called = True  # 🔥 标记
          
          else:
              # 基础 Tool
              result = tool_func(...)
          
          # 记录
          action_history.append((tool_call, result))
      
      # 5. ask_user 后立即触发 PlanAgent（保险机制）
      if ask_user_called:
          plan = plan_agent.replan(...)
          action_history = []
      
      # 6. 定期触发（兜底）
      elif turn % 10 == 0:
          plan = plan_agent.replan(...)
          action_history = []

plan 过期检测规则 (is_plan_outdated):
  • 用户反馈包含关键词: "不要"、"改成"、"换成"、"重新"、"取消"、"修改"
  • 实际执行与 plan 的 next_n_steps 严重偏离 (超过 50%)

决策指南 (在 System Prompt 中):
  • 用户需求清晰 + 标准流程 → use_*_workflow
  • 用户需求模糊 + 需要讨论 → call_* + ask_user
  • 需要迭代优化 → 逐步调用 SubAgent + ask_user
```

### 2. Trace System (状态树)
```
每个 Trace 节点包含:
├─ node_id (唯一标识)
├─ parent_id (父节点)
├─ state (状态快照)
├─ action (执行的动作: Tool/SubAgent/Orchestrator)
├─ result (执行结果)
└─ children (子节点列表)

Trace System 能力:
• 状态树管理 (树的不同分支 = 多方案)
• 多方案并行探索
• 回溯到任意历史节点
• 对比不同方案的效果
• 结合 LangGraph Checkpointer 持久化
```

### 3. 无限运行机制
```
PlanAgent 触发条件（三种情况）:

1. 定期触发: 每 10 步自动触发（兜底）
2. ask_user 触发: 用户反馈后立即触发（保险机制）
3. 主动检测: 每轮开始前检查 plan 是否过期（规则检测）

触发后执行:
1. PlanAgent 分析当前进度,重新规划
2. 固化计划信息到 thinking.json
3. 清空 action_history (释放上下文)
4. 更新 Trace 树

关键设计:
• 历史操作的效果在文件系统中,Agent 读文件了解历史
• ask_user 会导致 plan 失效（用户可能改变任务方向）
• 主动检测规则: 关键词匹配（"不要"、"改成"、"换成"等）

结合 LangGraph:
• LangGraph Checkpointer: 管理 Workflow 内部状态
• Trace System: 管理 Workflow 外部决策树
• trim_messages 压缩对话历史
```

### 4. Skills 动态加载
```
根据当前任务动态加载相关 Skills:
• 分析任务 → 识别需要的知识类型
• 查询 Skills 库 → 加载相关文档
• 组合 SubSkills → 构建完整上下文
• 注入到 Prompt → 减少无效 token
```

---

## 目录结构

```
akg_agents/
├── core_v2/                        # 通用框架 (新架构,不依赖算子)
│   ├── agents/
│   │   ├── base.py                 # AgentBase (100行)
│   │   └── registry.py             # @register_agent
│   ├── orchestrator/
│   │   ├── orchestrator.py         # Orchestrator (对StateGraph封装)
│   │   └── builder.py              # WorkflowBuilder (原BaseWorkflow)
│   ├── llm/
│   │   ├── provider.py             # LLMProvider 接口
│   │   └── client.py               # LLMClient
│   ├── workers/
│   │   └── manager.py              # WorkerManager (保持现有)
│   ├── state/
│   │   ├── trace_system.py         # Trace System (状态树)
│   │   ├── storage.py              # StateStorage
│   │   ├── plan_agent.py           # PlanAgent
│   │   └── compressor.py           # ActionCompressor
│   ├── tools/
│   │   ├── base_tools/             # 通用基础工具
│   │   │   ├── file_tools.py
│   │   │   └── ...
│   │   └── loader.py               # Domain Tools 动态加载
│   └── skills/
│       ├── loader.py               # 动态加载
│       └── manager.py              # Skills 管理
│
├── ops/                            # 算子专用层
│   ├── agents/
│   │   ├── kernel_agent.py         # KernelAgent (主Agent)
│   │   ├── designer.py             # Designer SubAgent
│   │   ├── coder.py                # Coder SubAgent
│   │   └── ...                     # 其他 SubAgent
│   ├── tools/
│   │   ├── verify_tool.py          # verifyTool (原KernelVerifier)
│   │   ├── compile_tool.py         # compileTool
│   │   ├── profile_tool.py         # profileTool
│   │   └── ...
│   ├── orchestrators/              # 工作流构建器(原workflows)
│   │   ├── standard_kernel.py      # 标准算子生成 (原DefaultWorkflow)
│   │   ├── evolve.py               # 优化迭代
│   │   └── search.py               # 方案搜索
│   └── skills/                     # Kernel Skills 系统
│       ├── compilers/              # 编译器知识
│       │   ├── cuda.md
│       │   ├── ascend.md
│       │   └── ...
│       ├── dsls/                   # DSL 知识
│       │   ├── triton.md
│       │   ├── swft.md
│       │   └── ...
│       ├── patterns/               # 优化模式
│       ├── examples/               # 示例代码
│       └── best_practices/         # 最佳实践
│
├── common/                         # 通用场景层 (NEW) - 与 ops/ 同级
│   ├── agents/                     # 通用场景 Agents
│   ├── tools/                      # 通用场景工具
│   └── skills/                     # Skills 管理
│       ├── (外部 Skills 加载接口)
│       └── (Skills 配置)
│
├── hitl/                           # HITL 编排层 (NEW)
│   └── (集成到 core_v2/state/ 和 ops/agents/、common/agents/)
│
├── core/                           # 现有代码 (逐步废弃)
│   └── ...
│
└── cli/                            # CLI 交互层 (重构)
    ├── commands/
    │   ├── op/                     # op 命令实现
    │   ├── common/                 # common 命令实现 (NEW)
    │   └── slash_commands.py       # 斜杠命令系统 (NEW)
    ├── ui/
    │   ├── completers.py           # 自动补全器 (NEW)
    │   └── panel_simple.py         # 简化面板 (NEW)
    └── runtime/
        └── ...                     # 保持现有
```

---

## 迁移路线

### 阶段1: 算子内容迁移 (2-3周)
```
1. 创建 ops/ 目录
2. 迁移所有算子专用 Agent
3. 迁移 KernelVerifier 和适配器
4. 迁移编译器和 DSL
5. 组织算子 Skills 知识库
```

### 阶段2: 核心框架重构 (2-3周)
```
1. 创建 core_v2/ 目录
2. 实现精简版 AgentBase
3. 实现 LLM 抽象层
4. 实现验证框架
5. 实现 Worker 系统
6. 实现 Tools 系统
7. 适配 LangGraph (复用现有机制)
```

### 阶段3: Orchestrator & Workflow (1-2周)
```
1. 实现 Orchestrator (对 StateGraph 封装)
2. BaseWorkflow → WorkflowBuilder (改名+迁移)
3. 实现 from_yaml() (复用现有 YAML 配置)
4. 现有 Workflows 迁移到 ops/orchestrators/
```

### 阶段4: HITL & Agent (2-3周)
```
1. 实现 PlanAgent (规划+固化)
2. 实现 KernelAgent (基于 ReAct)
3. 实现 Trace System (状态树管理)
4. Agent + Skills 动态组合机制
5. KernelAgent 集成 Workflow 调用
```

### 阶段5: Skills 化改造 (持续)
```
1. 所有文档 → Skills
2. 所有工具 → Tools
3. 所有能力 → Agents
4. 动态加载机制
```

### 阶段5.5: 通用场景层建设 (1-2周) (NEW)
```
1. 创建 common/ 目录结构 (与 ops/ 同级)
2. 实现外部 Skills 加载接口
3. 实现通用场景 Agents
4. akg_cli common 命令支持
5. 验证至少 1 个非算子场景
6. 证明 core_v2 的通用性
```

### 阶段6: CLI 交互层重构 (3-5天) (NEW)
```
1. 实现斜杠命令注册系统
2. 实现命令自动补全器
3. 移除 Top5 框架，简化面板
4. 实现核心斜杠命令 (/trace, /parallel, /help, /mode 等)
5. akg_cli common 命令支持（复用 op 的交互逻辑）
6. 用户体验测试和优化
```

### 阶段7: 整合测试 (1-2周)
```
1. 端到端测试 (算子场景 + 通用场景)
2. CLI 交互测试
3. 性能测试
4. 文档完善
```

---

## 成功标准

### 架构指标
- [ ] core_v2/ 完全不依赖 ops/
- [ ] 可用 core_v2/ 构建非算子应用
- [ ] AgentBase 精简到 100 行以内
- [ ] 充分复用 LangGraph 现有能力

### 能力指标
- [ ] 支持 Tools/SubAgents/Workflow 混合调用
- [ ] Orchestrator 编排器 (对 StateGraph 封装+YAML支持)
- [ ] WorkflowBuilder 模板 (原 BaseWorkflow,最小改动)
- [ ] Trace System: 状态树+多方案探索
- [ ] 无限运行: PlanAgent 每10步固化
- [ ] Skills 动态加载和组合
- [ ] 验证逻辑 Tools 化 (verifyTool等)
- [ ] akg_cli common 支持通用场景 (NEW)
- [ ] 外部 Skills 加载能力 (NEW)
- [ ] 至少验证 1-2 个非算子场景 (NEW)
- [ ] CLI 斜杠命令系统 (NEW)
- [ ] CLI 自动补全和历史记录 (NEW)
- [ ] CLI 简化面板（移除 Top5）(NEW)

### 质量指标
- [ ] 单元测试覆盖率 > 80%
- [ ] 端到端测试全部通过
- [ ] 性能无明显下降

---

## 待讨论问题

### 已明确 ✅
1. **Orchestrator 定位**: ✅ 编排器 (对 StateGraph 封装) + YAML 支持
2. **Workflow 定位**: ✅ 被编排出来的可执行流程 (compile 后的图)
3. **WorkflowBuilder**: ✅ 原 BaseWorkflow (改名),使用 Orchestrator 编排
4. **KernelAgent 决策**: ✅ 通过 Tool/SubAgent/Workflow 的 description/metadata 分析
5. **PlanAgent 触发**: ✅ 主要是每10步,其他时机暂不考虑
6. **向后兼容**: ✅ 不需要与 core 兼容,但充分复用现有代码 (StateGraph, YAML 等)

### 已设计 ✅
7. **Trace System vs Checkpointer**: ✅ Checkpointer 管 Workflow 内,Trace 管 Workflow 外
8. **Agent + Skills 组合**: ✅ SYSTEM_PROMPT + Skills (Markdown) 动态加载组合

### 待设计 🔧
9. **Skills 具体实现**: 【待实现】Markdown 格式、加载器、动态组合逻辑
10. **verifyTool 设计**: 【待设计】原 KernelVerifier Tools 化的具体方案

### 已设计 ✅
11. **CLI 交互层**: ✅ 斜杠命令系统、自动补全、简化面板（详见 `CLI_DESIGN.md`）

---

**更新时间**: 2026-01-25
**状态**: 架构设计阶段
**下一步**: 开始实施各模块
