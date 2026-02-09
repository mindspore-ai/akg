# demo vs akg_agents 融合分析报告

> 生成时间: 2026-02-09
> 分析范围: `demo/` (KernelBench 任务构建 Agent) vs `akg_agents/` (通用算子生成框架)

---

## 一、架构总览

### 1.1 demo/ 架构

```
demo/                           # ~3500 行 Python
├── main.py                     # CLI 入口 (181 行)
├── config.py                   # 配置 (53 行, 复用 akg_agents LLM)
├── agent/
│   ├── react_loop.py           # ReAct 循环 (669 行)
│   └── prompts.py              # 系统提示词 (129 行)
├── tools/
│   ├── registry.py             # 工具注册表 (75 行)
│   ├── code_tools.py           # 代码工具 (998 行, 核心)
│   ├── file_tools.py           # 文件工具 (655 行)
│   └── user_tools.py           # 用户交互 (33 行)
└── task/
    ├── input_parser.py         # 输入解析 (158 行)
    ├── task_builder.py         # 任务格式 (132 行)
    └── test_constructor.py     # 验证测试 (397 行)
```

**特点**: 专注 KernelBench 任务构建的垂直 Agent，自包含、轻量、高度专用。

### 1.2 akg_agents/ 架构

```
akg_agents/python/akg_agents/   # ~15000+ 行 Python
├── cli/                        # CLI 入口 (Typer, 交互式 TUI)
├── core/                       # v1 核心框架 (LangChain/LangGraph)
│   ├── agent/                  # Agent 基类 + 多种角色
│   ├── tools/                  # Pydantic Schema 工具系统
│   ├── skills/                 # 技能加载 v1
│   ├── sub_agent_registry.py   # 子 Agent 注册表 (1263 行)
│   └── task.py                 # 任务编排 (679 行)
├── core_v2/                    # v2 核心框架 (原生实现)
│   ├── agents/                 # ReAct Agent v2 (703 行)
│   ├── tools/                  # 统一 Tool Executor
│   ├── skill/                  # 技能系统 v2 (层级、版本、选择)
│   └── config/                 # 多级配置 (627 行)
├── op/                         # 算子领域
│   ├── workflows/              # LangGraph 工作流
│   ├── resources/skills/       # 63 个 SKILL.md
│   └── agents/verifier/       # 验证子 Agent
└── utils/                      # 工具库
```

**特点**: 通用框架，多 Agent 协作，支持 Skill 系统、工作流编排、多后端。

---

## 二、核心模块对比

### 2.1 工具系统 (Tools)

| 维度 | demo | akg_agents v1 | akg_agents v2 |
|------|------|---------------|---------------|
| **定义方式** | JSON Schema dict | Pydantic BaseModel | 函数签名 + YAML |
| **注册方式** | `ToolRegistry.register()` 手动注册 | `@tool()` LangChain 装饰器 | 模块级函数 + 动态发现 |
| **调用格式** | `{"action": "tool_name", "arguments": {...}}` | LangChain `tool_calls` | `{"tool_name": "...", "arguments": {...}}` |
| **执行方式** | `ToolRegistry.execute()` 同步 | LangChain `tool.invoke()` | `ToolExecutor.execute()` 异步 |
| **返回格式** | `{status, output, error}` | LangChain ToolMessage | `{status, output, error_information}` |
| **类型安全** | 运行时 (JSON Schema) | 编译时 (Pydantic) | 运行时 (函数签名) |
| **工具路由** | 直接查表 | LangChain 路由 | 类型路由 (Agent/Workflow/Domain/Basic) |
| **上下文传递** | 无 | LangChain state | `agent_context` dict |
| **Prompt 生成** | `list_for_prompt()` 文本列表 | LangChain tools 列表 | 函数内省 + YAML |

#### 评估

**demo 优势**:
- **简单直接**: 75 行 registry 实现完整功能，无框架依赖
- **JSON Schema 统一**: 定义和 LLM 看到的格式一致，调试方便
- **`list_for_prompt()` 灵活**: 可精确控制 LLM 看到的工具描述
- **领域专用工具强大**: `assemble_task`, `trace_dependencies` 等是 demo 的核心竞争力

**akg_agents 优势**:
- **类型安全 (v1)**: Pydantic 在编译时捕获参数错误
- **异步支持 (v2)**: 天然支持并发工具执行
- **工具路由 (v2)**: 支持将 Agent、Workflow 作为工具调用
- **上下文传递 (v2)**: 工具可访问硬件参数、执行历史等上下文

#### 结论

demo 的注册式设计更适合**专用 Agent**（工具集固定、描述精细），akg_agents v2 的类型路由适合**框架级编排**（Agent 嵌套、工作流组合）。

**融合建议**: 保留 demo 的 JSON Schema 注册模式（简单可控），但包装为 akg_agents v2 兼容的 `ToolExecutor` 插件。核心领域工具 (`assemble_task` 等) 作为 domain tools 注册。

---

### 2.2 ReAct 循环

| 维度 | demo | akg_agents v1 | akg_agents v2 |
|------|------|---------------|---------------|
| **循环结构** | `for` 循环 + 最大步数 | LangGraph 托管 | `while True` 异步 |
| **JSON 解析** | 4 策略容错（直接/代码块/大括号/修复） | LangChain 原生 | `extract_nested_json` |
| **历史管理** | 手动压缩（保留 workspace 引用） | `trim_messages` 中间件 | LLM 摘要压缩 |
| **重试逻辑** | 3 次重试 + 引导消息 | 框架托管 | 无（失败转 ask_user） |
| **最大步数** | 50（可配置） | 框架托管 | 无限制 |
| **流式输出** | 不支持 | 通过 CLI executor 支持 | 不支持 |
| **日志系统** | SessionLogger（文件级详细记录） | 标准 logging | TraceSystem（树结构） |
| **任务恢复** | 不支持 | LangGraph checkpointer | TraceSystem 恢复 |
| **调试难度** | 低（显式控制） | 高（框架黑盒） | 中 |

#### 评估

**demo 优势**:
- **JSON 解析容错**: 4 策略 fallback 是实战验证的关键能力，v2 只有单策略
- **历史管理精细**: 保留 workspace 引用不被压缩，对代码生成任务至关重要
- **SessionLogger 详尽**: 每步记录 thought/action/result，完美可回溯
- **重试+引导**: 失败时给 LLM 具体的修复指导（如 "JSON 太长请拆分"），而非简单重试

**akg_agents 优势**:
- **异步 (v2)**: 支持并发
- **任务恢复 (v2)**: TraceSystem 支持断点续行
- **LLM 摘要压缩 (v2)**: 比简单截断更智能
- **流式输出 (v1 CLI)**: 实时反馈

#### 结论

demo 的 ReAct 循环在**鲁棒性**上远优于 akg_agents v2（4 策略解析 + 重试引导 + 精细历史管理），但缺少异步和流式支持。

**融合建议**: 以 demo 的循环为骨架，加入 v2 的异步支持和 TraceSystem。

---

### 2.3 Prompt/模板系统

| 维度 | demo | akg_agents |
|------|------|------------|
| **格式** | Python 字符串模板 | Jinja2 `.j2` 文件 |
| **工具描述注入** | `{tools_description}` 占位符 | LangChain tools 列表 / Jinja2 |
| **Skill 注入** | 无 | Jinja2 循环渲染 |
| **工作流指导** | 硬编码在 SYSTEM_PROMPT 中 | 拆分到 SKILL.md 中 |
| **灵活性** | 低（改代码才能改 prompt） | 高（改文件即可） |
| **可读性** | 高（集中查看） | 中（分散在多个文件） |

#### 结论

demo 的 prompt 过于硬编码，但内容密度高且经过实战打磨。应迁移为 Jinja2 模板 + SKILL.md 组合。

---

### 2.4 Skill 系统

akg_agents 的 Skill 系统（尤其 v2）是一个成熟的知识管理框架：

```
Skill 系统 v2 特性:
├── 多路径发现 (项目级 + 全局)
├── YAML 元数据 (level, category, version, backend...)
├── 层级结构 (parent → children, exclusive groups)
├── 两阶段选择 (粗筛 metadata → LLM 精选)
├── 63 个已有 Skill (triton-cuda, triton-ascend, tilelang...)
└── 与 Agent prompt 深度集成
```

**demo 能否成为 Skill?**

**结论: 完全可行且强烈推荐**

demo 的工作流天然适合 SKILL.md 格式：
- 当前 `prompts.py` 中的 SYSTEM_PROMPT（KernelBench 格式 + 工作流步骤 + 规则）→ 拆分为 SKILL.md 内容
- 当前的 `tools/code_tools.py` 中的领域工具 → 保留为独立工具
- 当前的 `task/test_constructor.py` → 可打包为 scripts/

详细方案见第三节。

---

## 三、Skill 化方案

### 3.1 Skill 结构设计

```
kernelbench-task-builder/
├── SKILL.md                        # 主 Skill 定义
├── references/
│   ├── kernelbench-format.md       # KernelBench 格式规范
│   ├── workflow-steps.md           # 详细工作流步骤
│   └── assembly-strategies.md      # 任务装配策略
└── scripts/
    └── validate_task.py            # 任务验证脚本
```

### 3.2 SKILL.md 内容

```yaml
---
name: kernelbench-task-builder
description: >
  从 PyTorch/Triton 代码仓中提取算子实现，构建为 KernelBench 标准格式的
  单文件自包含任务。支持代码提取、依赖追踪、函数内联、格式验证。
level: L1
category: workflow
version: "1.0.0"
metadata:
  task_type: code_transformation
  target_format: kernelbench
  input_types: code,file,directory
---
```

核心内容从 `prompts.py` SYSTEM_PROMPT 迁移，包括：
- KernelBench 格式规范
- 5 步工作流（分析 → 依赖追踪 → 策略选择 → 装配 → 验证）
- 3 条核心规则（禁止重写/保留返回值/不确定时ask）
- 工具使用指南

### 3.3 迁移路径

| 当前位置 | 迁移目标 | 说明 |
|---------|---------|------|
| `prompts.py` SYSTEM_PROMPT | `SKILL.md` + `references/` | 工作流知识 |
| `code_tools.py` 领域工具 | `fused/tools/` → domain tools | 保留为工具 |
| `file_tools.py` 基础工具 | 复用 akg_agents basic_tools | 已有等价实现 |
| `test_constructor.py` | `scripts/validate_task.py` | 验证脚本 |
| `react_loop.py` | 复用 akg_agents v2 ReAct + 增强 | Agent 执行框架 |
| `config.py` | 复用 akg_agents settings | 配置系统 |

---

## 四、工具系统深度分析

### 4.1 demo 独有工具（核心竞争力）

这些工具是 demo 的核心价值，akg_agents 中没有等价实现：

| 工具 | 功能 | 行数 | 核心技术 |
|------|------|------|---------|
| `assemble_task` | 从源文件装配 KernelBench 任务 | ~200 | AST 提取 + import 清理 + 排除/选择模式 |
| `trace_dependencies` | 函数依赖追踪 + 外部调用检测 | ~80 | AST import 分析 + BFS + 私有模块检测 |
| `validate_task` | 任务格式验证 | ~30 | TestConstructor 子进程验证 |
| `test_with_reference` | 参考对比测试 | ~50 | 多组输入 + per-case init_inputs |
| `read_function` | AST 函数提取 + workspace 持久化 | ~60 | AST 精确定位 + 自动保存 |
| `grep_search` | 正则搜索 | ~40 | 多文件搜索 + 上下文 |

### 4.2 可复用的 akg_agents 工具

| demo 工具 | akg_agents 等价 | 差异 |
|-----------|----------------|------|
| `read_file` | `read_file` (basic_tools) | demo 支持 offset/limit 分页，更灵活 |
| `write_file` | `write_file` (basic_tools) | 基本等价 |
| `run_code` | `execute_script` (basic_tools) | demo 支持内联 code 字符串 |
| `ask_user` | `ask_user` (basic_tools) | akg_agents 用 LangGraph interrupt，更优 |

### 4.3 融合工具注册方案

```python
# 方案: 将 demo 工具包装为 akg_agents v2 兼容格式

# 1. 保留 demo 的 ToolRegistry 作为"内部注册表"
# 2. 暴露为 akg_agents v2 的 domain tools
# 3. 通过 adapter 桥接执行
```

---

## 五、ReAct 循环深度分析

### 5.1 demo 独有能力

1. **多策略 JSON 解析** (价值: ★★★★★)
   ```
   策略1: 直接 json.loads()
   策略2: 提取 ```json 代码块
   策略3: 提取最外层 {...} 大括号
   策略4: 修复常见错误（尾逗号/单引号/截断检测）
   ```
   这是 demo 最重要的鲁棒性保障。LLM 经常输出格式不完美的 JSON。

2. **workspace 引用保护** (价值: ★★★★)
   历史压缩时保留 workspace 文件引用，避免 Agent 丢失已提取代码的位置。

3. **截断检测+引导** (价值: ★★★★)
   检测到 LLM 输出被截断时，主动引导 "代码太长，请拆分或使用工具"。

4. **工具结果格式化** (价值: ★★★)
   统一的 `状态/输出/错误` 格式，LLM 容易理解。

### 5.2 akg_agents v2 独有能力

1. **TraceSystem** (价值: ★★★★)
   树结构追踪，支持断点续行。

2. **异步执行** (价值: ★★★)
   async/await 天然支持并发。

3. **LLM 摘要压缩** (价值: ★★★)
   用 LLM 总结历史，比截断更智能（但有延迟成本）。

### 5.3 融合方案

以 demo 的循环为骨架，融入 v2 的优势：
- 保留 4 策略 JSON 解析
- 保留 workspace 引用保护
- 加入异步支持
- 加入 TraceSystem
- 可选 LLM 摘要压缩

---

## 六、融合方案总结

### 6.1 四种方案对比

| 方案 | 工作量 | 风险 | 收益 | 推荐度 |
|------|--------|------|------|--------|
| **A: 直接合入** demo → akg_agents | 低 | 中 | 低 | ★★ |
| **B: 按 akg_agents 重写 demo** | 高 | 高 | 中 | ★★ |
| **C: 修改 akg_agents 适配 demo** | 中 | 中 | 中 | ★★★ |
| **D: 融合两者** | 中 | 低 | 高 | ★★★★★ |

### 6.2 推荐方案: D (融合)

```
融合策略:
                    akg_agents 框架
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Skill 系统      v2 ReAct Agent    Tool Executor
         │               │               │
         │      ┌────────┼────────┐      │
         │      │  demo 增强      │      │
    ┌────┴────┐ │  · 4策略JSON解析│ ┌────┴────┐
    │ kernel  │ │  · workspace管理│ │ demo    │
    │ bench   │ │  · 重试+引导   │ │ domain  │
    │ task    │ │  · 截断检测    │ │ tools   │
    │ builder │ └────────────────┘ │         │
    │ SKILL   │                    │assemble │
    │ .md     │                    │trace    │
    └─────────┘                    │validate │
                                   │test_ref │
                                   └─────────┘
```

**具体步骤**:

1. **Skill 化**: 将 demo 的工作流知识打包为 `kernelbench-task-builder` SKILL.md
2. **工具适配**: 将 demo 的领域工具注册为 akg_agents v2 的 domain tools
3. **ReAct 增强**: 将 demo 的鲁棒性特性 (JSON 解析/历史管理) 贡献回 v2 ReAct
4. **Agent 注册**: 创建 `KernelBenchAgent` 作为 v2 的 specialized agent

### 6.3 实现优先级

| 步骤 | 内容 | 工作量 |
|------|------|--------|
| P0 | Skill 化 (SKILL.md + references) | 1 天 |
| P1 | 工具适配层 (adapter.py) | 2 天 |
| P2 | KernelBenchAgent (复用 v2 ReAct + demo 增强) | 3 天 |
| P3 | ReAct v2 增强 (贡献 JSON 解析等) | 2 天 |
| P4 | 完整集成测试 | 1 天 |

---

## 七、关键风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| v2 ReAct 缺少 demo 的鲁棒性 | JSON 解析失败率上升 | 优先 P3，将 4 策略解析贡献回 v2 |
| Skill 过大导致 prompt 超长 | LLM 效果下降 | 拆分为 references/，按需加载 |
| 工具接口不兼容 | 集成困难 | adapter 层隔离，渐进迁移 |
| demo 工具依赖文件系统路径 | 在 akg_agents 环境下路径错误 | 统一 path resolver |
