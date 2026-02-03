# AKG_CLI Common 文档（ReAct 通用模式）

> 说明：本文 **只覆盖 `akg_cli common`**。不包含其它子命令的任何内容。

---

## 目录

- [1. 介绍与定位](#1-介绍与定位)
- [2. 快速开始](#2-快速开始)
- [3. 交互与斜杠命令](#3-交互与斜杠命令)
- [4. 运行时架构（实现细节）](#4-运行时架构实现细节)
- [5. ReAct 执行与状态流](#5-react-执行与状态流)
- [6. 工具体系（工具清单 + 参数 + 行为）](#6-工具体系工具清单--参数--行为)
- [7. 权限与安全策略](#7-权限与安全策略)
- [8. 计划模式与 Todo](#8-计划模式与-todo)
- [9. 上下文记忆与压缩](#9-上下文记忆与压缩)
- [10. 调试与可观测性](#10-调试与可观测性)
- [11. 示例：SWE-bench 单样本跑通](#11-示例swe-bench-单样本跑通)
- [12. 扩展指南（从实现到扩展）](#12-扩展指南从实现到扩展)
- [13. 边界与常见问题](#13-边界与常见问题)

---

## 1. 介绍与定位

`akg_cli common` 是 **通用场景的 ReAct CLI 入口**，适用于任意开发/运维/文档/调试任务，不要求指定框架、后端或硬件配置。

核心特性：

- **ReAct 循环**：Think → Action → Observation，允许调用工具完成任务。
- **无需目标配置**：不需要 `framework/backend/arch/dsl` 等参数。
- **工具驱动**：工具集合从 `akg_agents/python/akg_agents/tool/*.txt` 动态加载。
- **可交互/可脚本化**：支持交互式对话，也支持 `--intent --once` 一次性执行。
- **工具审批**：高风险工具需要用户确认，可用 `--yolo` 自动批准。
- **上下文压缩**：自动/手动压缩历史上下文，保持长会话可用。

> 运行范围：`common` 仅使用 **本地执行链路**（`LocalExecutor`），不依赖外部服务。

---

## 2. 快速开始

### 2.1 启动交互式会话

在你想操作的工作目录下执行（该目录将作为 `common` 的“工作区根”）：

```bash
akg_cli common
```

进入提示符：

```
common>
```

输入需求即可开始多轮对话。

### 2.2 单次请求（非交互）

```bash
akg_cli common --intent "请分析该目录并生成一份 README" --once
```

- `--intent`：直接提供请求内容
- `--once`：处理完一次请求后退出（适合脚本/批处理）

### 2.3 关键命令行参数

- `--intent <文本>`：跳过输入循环，直接执行该请求
- `--once/--exit-after-intent`：执行完 `--intent` 后退出
- `--stream/--no-stream`：是否启用流式输出（默认 `--stream`）
- `--yolo`：自动批准所有需审批的工具执行

> 注意：`common` 的**相对路径解析基于当前工作目录**，建议从仓库根目录启动。

---

## 3. 交互与斜杠命令

`common` 使用 `prompt_toolkit` 的交互循环，支持斜杠命令（`/` 开头）：

- `/help`：查看命令帮助（支持 `Tab` 自动补全）
- `/compact`：手动压缩上下文（common 专用）
- `/list_tools`：列出当前可用工具（common 专用）
- `/display_last_raw_llm_input`：显示上一次 LLM 原始输入（common 专用）
- `/exit`（别名 `/quit`, `/q`）：退出

通用退出方式：

- 输入 `exit` / `quit` / `:q`
- 或使用 `/exit`

> 斜杠命令有“场景隔离”（`scene='common'`）。部分命令仅在其它场景可用，common 中会提示不可用。

---

## 4. 运行时架构（实现细节）

### 4.1 关键入口与文件

- CLI 入口：`akg_agents/python/akg_agents/cli/cli.py`
- Common 命令：`akg_agents/python/akg_agents/cli/commands/common.py`
- 本地执行器：`akg_agents/python/akg_agents/cli/runtime/local_executor.py`
- Common ReAct 执行器：`akg_agents/python/akg_agents/cli/runtime/common_executor.py`
- 工具层：`akg_agents/python/akg_agents/cli/runtime/common_tools.py`
- 工具描述：`akg_agents/python/akg_agents/tool/*.txt`

### 4.2 端到端调用链路

```
用户输入
  ↓
CommonCommandRunner (common.py)
  ↓
LocalExecutor.execute_common_agent()
  ↓
CommonTurnExecutor.run_turn()
  ↓
LangChain create_agent() + LangGraph astream(updates)
  ↓
工具调用/模型输出
  ↓
MessageSender → AKGConsole
```

### 4.3 关键职责划分

- **CommonCommandRunner**：输入循环、斜杠命令调度、退出逻辑
- **LocalExecutor**：会话管理、消息路由、创建 CommonTurnExecutor
- **CommonTurnExecutor**：ReAct 执行、工具调度、上下文管理、压缩
- **ToolRunner**：具体工具实现（读写、bash、patch、grep 等）

---

## 5. ReAct 执行与状态流

### 5.1 ReAct 主循环

`common` 使用 LangGraph 的 `agent.astream(..., stream_mode="updates")` 事件流，处理三类事件：

- `model`：LLM 输出（文本 + reasoning）
- `tools`：工具执行结果
- `__interrupt__`：工具向用户提问（`ask_user`）时的暂停/恢复

### 5.2 状态字段（返回给 CLI）

执行结果是一个 dict，常见字段：

- `current_step`：`react` / `completed` / `waiting_for_user_input` / `cancelled_by_user` / `error`
- `should_continue`：是否继续当前会话（common 默认允许继续）
- `display_message`：展示给用户的内容
- `hint_message`：辅助信息（非流式时通常是 reasoning）
- `auto_input`：计划模式切换时的自动输入（最多 3 次自动 hop）

### 5.3 交互细节

- **流式模式**：模型输出会实时流式打印（`LLMStreamMessage`），工具结果用 `DisplayMessage` 面板展示。
- **非流式模式**：模型输出被收集，最终一次性展示；若包含 `final_answer` JSON 则会自动截取。
- **完成后继续**：当 `current_step=completed` 时，CLI 仍允许继续输入新需求，并会显示“本轮任务已完成”的提示。
- **取消处理**：`Ctrl+C` 触发取消，`LocalExecutor.cancel_main_agent` 会尝试取消当前任务并回滚未完成的 tool call。
- **工具输出格式**：工具返回会以 `[tool:<name>]` 标识；除 `edit/multiedit/write/apply_patch` 外还会附带格式化参数行。
- **错误展示**：当 `current_step=error` 且包含错误字段时，CLI 会打印 `Error:` 提示。

---

## 6. 工具体系（工具清单 + 参数 + 行为）

### 6.1 工具加载机制

- 工具描述来自：`akg_agents/python/akg_agents/tool/*.txt`
- 运行时由 `ToolDocs` 读取并生成 `StructuredTool`
- 参数 schema 采用 `GENERIC_ARGS_SCHEMA`（允许任意字段）
- 未实现的工具会返回：`[ERROR] Tool '<name>' is not implemented in akg_cli common.`
- diff 输出会尝试走 `delta`/`git-delta` 渲染；若未安装则输出原始 diff

> **重要差异**：工具描述文本中“必须绝对路径”的要求，在 `common` 中被扩展为“支持相对路径”。

### 6.2 路径解析与工作区

- **工作区根** = 启动 `akg_cli common` 时的 `cwd`
- 相对路径会自动解析为 `cwd` 下路径
- **强制在工作区内的工具**：`read` / `write` / `edit` / `multiedit` / `apply_patch`
- **允许跨目录的工具**：`ls` / `glob` / `grep`（不做 workspace 校验）
- **bash**：通过解析命令检测外部路径并询问用户是否放行

### 6.3 工具清单（逐条行为说明）

以下工具均可在 `common` 中使用：

#### 1) `read`

- 作用：读取文件（返回带行号的文本）
- 关键参数：
  - `filePath` / `file_path` / `path`
  - `offset`（行偏移，默认 0）
  - `limit`（行数限制）
- 行号从 1 开始；每行最长 2000 字符
- **限制**：路径必须存在且为文件，且在工作区内
- 空结果返回 `[EMPTY]`

#### 2) `write`

- 作用：写入/覆盖文件
- 参数：`filePath` + `content`
- 支持新建文件（自动创建父目录）
- **限制**：必须在工作区内
- **需要审批**（可被 `--yolo` 自动通过）
- 输出为 unified diff（若安装 `delta` 则彩色显示）
- 若无变化会返回 `[INFO] write: no changes ...`

#### 3) `edit`

- 作用：精确字符串替换
- 参数：`filePath`, `oldString`, `newString`, `replaceAll`
- **限制**：文件存在且在工作区内
- **需要审批**
- 若 `oldString` 找不到或多次出现且未 `replaceAll`，返回错误

#### 4) `multiedit`

- 作用：同一文件多处替换（原子性）
- 参数：`filePath`, `edits=[{oldString,newString,replaceAll}]`
- **限制**：文件存在且在工作区内
- **需要审批**
- 任一 edit 失败会整体失败（原子性）

#### 5) `apply_patch`

- 作用：补丁应用（支持两种格式）
  - **Codex Patch**：`*** Begin Patch` / `*** Update File` / `*** End Patch`
  - **Unified Diff**：走系统 `patch -p0 -u -`
- **限制**：所有涉及文件必须在工作区内
- **需要审批**
- 输出为差异 diff；若 `patch` 命令缺失会报错
- Codex Patch 仅支持 Add/Update/Delete（不支持 Move/Rename）

#### 6) `ls`

- 作用：列目录
- 参数：`path`（可选）、`ignore`（glob 模式列表）
- 返回目录项（目录以 `/` 结尾）
- 路径不存在或非目录会报错

#### 7) `glob`

- 作用：按 glob 模式查找文件
- 参数：`pattern`（必填）、`path`（可选）
- 结果按修改时间倒序排序
- 返回路径为绝对路径字符串

#### 8) `grep`

- 作用：正则搜索文件内容
- 参数：`pattern`（必填）、`include`（文件模式）、`path`
- 返回格式：`path:line:content`
- `path` 既可为目录也可为文件

#### 9) `bash`

- 作用：执行 shell 命令
- 参数：
  - `command`（必填）
  - `workdir`/`cwd`（可选）
  - `timeout`（默认 120 秒，传入 >1000 视为毫秒）
- **路径安全解析**：使用 tree-sitter-bash 解析命令，检测路径操作
  - 若解析失败：审批类型变为 `bash:unparsed`
  - 若发现外部路径或动态路径：审批类型为 `bash:external_paths`
- 检测的“路径命令”包括：`cd/rm/cp/mv/mkdir/rmdir/touch/chmod/chown/chgrp/ln/install`
- 输出包含退出码：
  - `[exit_code=0]` + stdout/stderr

#### 10) `question`

- 作用：向用户提问（ask_user）
- 参数：`question`/`prompt`/`message`，`options`，`custom`
- 返回用户回复（由 `ask_user` 中断/恢复机制提供，通常以“用户回复:”开头）

#### 11) `todowrite`

- 作用：写入 Todo 列表（会话内）
- 参数：`todos`/`items`/`tasks` 或 JSON 字符串
- 自动补齐字段：`content`, `status`, `priority`, `id`
- 返回 JSON 格式的 todo 列表
- JSON 解析失败会返回错误

#### 12) `todoread`

- 作用：读取当前 Todo 列表
- 无参数，返回 JSON

#### 13) `plan-enter`

- 作用：请求进入计划模式
- 会提示用户确认
- 确认后切换为 `plan` 模式（系统 prompt 变化）

#### 14) `plan-exit`

- 作用：请求退出计划模式并进入执行模式
- 会提示用户确认
- 确认后切换为 `build` 模式并记录 `plan.md` 路径

#### 15) `webfetch`

- 作用：拉取 URL 内容（httpx）
- 参数：`url`/`link`
- 返回最多 8000 字符（过长会截断）

#### 16) `batch`

- 作用：在一次调用中执行多条工具调用
- 参数：`payload` / `calls` / `items` / `batch`
- 每条调用形如：`{"tool": "read", "parameters": {...}}`
- 禁止嵌套 batch
- 返回每个调用的结果列表（JSON）
- 同一 batch 内按顺序执行；每条调用仍遵循各自的审批规则

---

## 7. 权限与安全策略

### 7.1 工具审批

以下工具默认 **需要审批**：

- `write`, `edit`, `multiedit`, `apply_patch`

审批由 `request_tool_approval()` 触发，使用 `ask_user` 进行确认。

### 7.2 `--yolo` 自动批准

启用 `--yolo` 后，会自动批准所有需要审批的工具（通过 ContextVar `tool_auto_approve` 实现）。

### 7.3 bash 的路径安全策略

- 使用 `tree_sitter` + `tree_sitter_languages` 解析 bash 命令
- 会扫描命令参数与重定向路径
- 如果发现外部路径或动态路径，进入增强审批（`bash:external_paths`）
- 如果解析器不可用，仍可执行，但会提示“无法检测外部路径”

---

## 8. 计划模式与 Todo

### 8.1 计划模式（Plan Mode）

- `plan-enter` 进入计划模式
- `plan-exit` 退出计划模式
- 计划文件默认：`<cwd>/plan.md`

进入计划模式后，系统提示将附加到 prompt：

- **禁止编辑文件 / 禁止 bash**（提示层约束）
- 推荐使用 `todowrite` 输出计划
- 完成后调用 `plan-exit` 切回 build 模式

> 计划模式是 **prompt 级约束**，并不强制限制工具调用。

### 8.2 Todo 系统

`CommonToolState` 中维护 `todos` 列表，`todoread/todowrite` 仅在当前会话有效：

- `todowrite` 自动规范化 todo 项：`status/priority/id`
- `todoread` 返回当前 todo 列表

---

## 9. 上下文记忆与压缩

### 9.1 记忆机制

- 使用 `create_checkpointer()` 保存会话历史（默认 memory backend）
- 使用 `trim_messages` 控制最大消息数（默认 100）

### 9.2 自动压缩

- 默认启用 `enable_auto_compact=True`
- 阈值：`max_messages * auto_compact_ratio`（默认 100 * 0.8 = 80）
- 达到阈值后，调用 LLM 生成摘要并替换历史
- 压缩摘要会作为下一轮的“历史摘要消息”（assistant role）注入
- 压缩完成后会重置 `thread_id`（`session_id` + 随机后缀），避免旧历史污染

### 9.3 手动压缩

- 通过 `/compact` 触发 `compact_history()`
- 成功会输出摘要面板

---

## 10. 调试与可观测性

### 10.1 last_raw_llm_input

`/display_last_raw_llm_input` 会输出一个结构化 JSON，包含：

- system_prompt（含技能元数据）
- model_preset（如 `deepseek_r1_default`）
- tools 列表（name/description）
- invoke_input（本轮 messages 或 resume）
- thread_id

### 10.2 Context Token 输出

每次模型输出后会打印：

```
[INFO] Context tokens: in=..., out=..., total=...
```

如果模型未提供 usage，显示 `N/A`。

---

## 11. 示例：SWE-bench 单样本跑通

仓库内提供完整示例：

- `akg_agents/examples/swebench_common_one.py`

功能流程：

1. 加载 SWE-bench Lite 数据集
2. clone 并 checkout base commit
3. 以 **子进程方式运行 `akg_cli common`**（`--intent --yolo --once`）
4. 产出 `predictions.jsonl`
5. 可选调用 `swebench.harness.run_evaluation`

典型调用：

```bash
python akg_agents/examples/swebench_common_one.py \
  --instance-id pallets__flask-4045 \
  --workdir workspace/swebench_runs \
  --stream
```

> 该示例是 `common` 的程序化集成参考模板。

---

## 12. 扩展指南（从实现到扩展）

### 12.1 新增工具

1. **增加工具描述文件**：
   - `akg_agents/python/akg_agents/tool/<tool>.txt`
2. **实现工具逻辑**：
   - 在 `common_tools._build_adapter_map()` 中注册
   - 在 `ToolRunner` 中添加实现
3. **如需审批**：将工具名加入 `_tool_requires_approval()`
4. **如需路径校验**：复用 `WorkspacePaths.ensure_within()`

> 若只加 `.txt` 而不实现，工具会返回“未实现”错误。

### 12.2 新增 Skills（common 专用）

- 目录：`akg_agents/python/akg_agents/cli/resources/skills/<skill>/SKILL.md`
- `CommonTurnExecutor` 会把技能元数据注入到 system prompt
- 格式示例（由 `build_common_skills_metadata` 生成）：
  - `- **skill_name**: 描述`
  - `  → read(filePath="...")`

> 如果 `cli/resources/skills` 不存在或为空，则不会注入任何技能。

### 12.3 自定义系统 Prompt

- 修改 `common_constants.py`：
  - `COMMON_SYSTEM_PROMPT`
  - `PLAN_MODE_SUFFIX`
  - `COMPACTION_SYSTEM_PROMPT`
  - `COMPACTION_USER_PROMPT`

### 12.4 模型与配置扩展

- 默认模型 preset：`deepseek_r1_default`（`LocalExecutor._load_common_config`）
- 可在 `core/llm/llm_config.yaml` 中新增 preset
- `CommonTurnExecutor` 支持的配置项（可在 `LocalExecutor` 注入）：
  - `agent_model_config.default`
  - `enable_auto_compact` / `auto_compact_ratio`
  - `max_messages`
  - `enable_memory` / `memory_backend`
  - `enable_trim`
  - `compaction_system_prompt` / `compaction_user_prompt`

### 12.5 扩展 Slash Commands

- 在 `cli/commands/slash_commands.py` 里注册：
  - `scene='common'`
  - 配置 `usage/examples/category`
- `CommonCommandRunner` 会自动接入 dispatcher

---

## 13. 边界与常见问题

**Q1：为什么读写文件提示“path outside workspace”？**

`read/write/edit/multiedit/apply_patch` 都强制限制在启动 `akg_cli common` 的工作目录内。请从目标仓库根目录启动，或使用正确的相对路径。

**Q2：bash 可以操作工作区外路径吗？**

可以，但会被 tree-sitter 检测到并要求显式审批。解析器不可用时会使用 `bash:unparsed` 审批。

**Q3：为什么工具说明里写“必须绝对路径”，common 却可以用相对路径？**

`common` 在运行时会对工具描述追加说明，允许相对路径并以 `cwd` 解析（见 `_augment_tool_desc`）。

**Q4：计划模式是否强制禁止工具调用？**

不强制。计划模式是 prompt 约束，需要靠 agent 遵守。

**Q5：Todo 是否会持久化？**

不会。`todos` 只存活于当前会话内存。

**Q6：apply_patch 为什么失败？**

- Codex Patch 必须有 `*** Begin Patch` / `*** Update File` 等头部
- Unified Diff 需要系统 `patch` 命令（`patch -p0 -u -`）

---

**至此，`akg_cli common` 的介绍、实现与扩展已覆盖完整。**
