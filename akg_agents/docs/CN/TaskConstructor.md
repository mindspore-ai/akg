# TaskConstructor - 标准化任务构建 Agent

## 概述

TaskConstructor 是一个自包含的 ReAct Agent，用于从现有 PyTorch/Triton 代码中提取算子实现，构建为 KernelBench 格式的标准化单文件任务。

**工具名**: `call_task_constructor`  
**Agent 类**: `TaskConstructor`  
**模块路径**: `akg_agents.op.agents.task_constructor`

### 与 OpTaskBuilder 的区别

| 维度 | TaskConstructor | OpTaskBuilder |
|------|----------------|---------------|
| 工具名 | `call_task_constructor` | `call_op_task_builder` |
| 输入 | 代码仓路径/文件/代码片段 | 纯文字需求描述 |
| 工作方式 | 内部 ReAct 循环 + AST 分析 + 17 个专用工具 | 单轮 LLM 生成 + 检查重试 |
| 适用场景 | **已有代码**，需要提取并标准化 | **没有代码**，只有文字描述 |

## 输入输出

### 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_input` | string | 是 | 用户需求描述 |
| `source_path` | string | 否 | 源代码路径（文件或目录） |

### 输出

| 字段 | 说明 |
|------|------|
| `status` | `"success"` 或 `"fail"` |
| `output` | 紧凑摘要（给外部 agent 看） |
| `task_code` | 完整 KernelBench 格式代码 |
| `task_code_path` | 输出文件绝对路径 |
| `op_name` | 算子名称 |
| `summary` | 构建过程摘要 |

## 工作流程

```
用户输入 → system_prompt + 工具集 → ReAct 循环
                                        │
    ┌───────────────────────────────────┘
    │
    ├─ 1. grep_search: 在源码中搜索目标函数
    ├─ 2. copy_to_workspace: 复制源文件到工作区
    ├─ 3. read_function: 提取目标函数代码
    ├─ 4. trace_dependencies: AST 依赖追踪
    ├─ 5. read_function: 提取外部依赖函数并内联
    ├─ 6. assemble_task: 构建自包含任务文件
    ├─ 7. validate_task → optimize_task → validate_task
    ├─ 8. test_with_reference: 多组输入对比验证
    └─ 9. finish: 返回结果
```

## 工具集详解（17 个）

### 代码分析与装配工具

#### `trace_dependencies`

AST 依赖追踪工具。给定源文件和入口函数名，通过 AST 解析自动发现该函数直接或间接调用的所有同文件函数，同时识别通过 import 别名引用的外部模块调用，并标注来源模块路径。返回完整的依赖链列表、建议嵌入的函数列表、以及需要手动处理的外部依赖。

- **参数**: `file_path`（workspace 中的文件路径）、`entry_functions`（入口函数名列表）
- **实现**: 基于 `ast_utils.py` 中的 `trace_function_deps()`，解析 import 别名映射，递归追踪函数调用链

#### `assemble_task`

选择性拼装工具。从 workspace 源文件中提取指定函数 + Model 类 + get_inputs → 组装自包含任务文件。支持三种用法：完整嵌入（整个文件）、选择性提取（指定函数列表）、排除式嵌入（排除不需要的函数）。工具内部使用 AST 解析精确提取/排除函数，自动包含文件头 import，自动清理装饰器和未使用的 import。

- **参数**: `source_files`（源文件列表）、`model_code`、`get_inputs_code`、`get_init_inputs_code`、`imports_code`（可选额外 import）、`helper_code`（可选辅助代码）
- **实现**: 调用 `ast_utils.py` 的 `extract_functions()` 进行精确提取，调用 `code_cleanup.py` 的 `cleanup_task_code()` 做基础清理

#### `validate_task`

预运行验证工具。对任务代码执行四步验证：1) 实例化 Model 类；2) 调用 forward 执行前向计算；3) 检查输出中是否存在 NaN/Inf；4) 多次运行检查一致性。通过 subprocess 在隔离环境中执行，超时自动终止。

- **参数**: `task_code`（代码字符串）或 `task_file`（文件路径）、`timeout`（超时秒数，默认 60）
- **实现**: 生成验证脚本写入临时文件，通过 `subprocess.run()` 执行，解析 stdout/stderr 判断结果

#### `optimize_task`

代码优化清理工具。在 validate 通过后、finish 前调用。执行：去重 import（合并重复的 `from X import a, b`）、移除私有模块引用（含 `_` 前缀的内部模块 import）、清理未使用的代码、格式化。优化后会自动再次验证确保未破坏代码。

- **参数**: `task_file`（文件路径）或 `task_code`（代码字符串）
- **实现**: 调用 `code_cleanup.py` 的 `polish_task_code()` 和 `merge_from_imports_text()` 进行文本和 AST 级别的清理

#### `test_with_reference`

正确性对比验证工具。将 Model 的输出与 reference 函数对比，支持多组输入。用户需要提供 `reference_code`（定义 `reference_forward(inputs, init_inputs)` 函数）和可选的 `multi_inputs_code`（定义 `get_multi_test_inputs()` 返回多组测试用例）。工具自动保存 reference 和 multi_inputs 代码到输出目录，方便复现。

- **参数**: `reference_code`（必填）、`task_file` 或 `task_code`、`multi_inputs_code`（可选）、`timeout`
- **实现**: 生成完整的对比测试脚本，通过 subprocess 执行，逐组比较输出的数值差异（allclose）

### 代码执行工具

#### `run_code`

Python 代码运行工具。传入 `code`（代码字符串）或 `file_path`（Python 文件路径），在隔离的 subprocess 中执行。捕获 stdout/stderr 并返回。workspace 目录作为工作目录，确保文件引用正确。

- **参数**: `code` 或 `file_path`（至少一个）、`timeout`（默认 30 秒）
- **实现**: 代码字符串写入临时文件后通过 `subprocess.run()` 执行

#### `apply_patch`

文件修改工具。查找 `old_string` 并替换为 `new_string`。如果 `old_string` 为空，则创建新文件写入 `new_string`。支持 workspace 路径前缀自动解析。

- **参数**: `file_path`、`old_string`（要替换的文本）、`new_string`（替换后文本）
- **实现**: 读取文件内容，执行字符串替换，写回文件

### 文件操作工具

#### `read_file`

文件读取工具。读取指定文件的内容，支持按行范围读取（`start_line`、`end_line`）。对超长文件自动截断到 300 行并提示使用行范围。支持 workspace 和外部路径。

- **参数**: `file_path`、`start_line`（可选）、`end_line`（可选）

#### `write_file`

文件写入工具。将内容写入指定文件，自动创建父目录。支持 workspace 路径前缀。

- **参数**: `file_path`、`content`

#### `append_to_file`

文件追加工具。在文件末尾追加内容（自动加换行分隔），适合分段写入大文件。

- **参数**: `file_path`、`content`

#### `scan_dir`

目录扫描工具。列出目录结构，显示文件名和大小，支持限制深度和最大文件数。对代码文件（`.py`）附加行数信息。

- **参数**: `dir_path`、`max_depth`（默认 3）、`max_files`（默认 100）

#### `copy_to_workspace`

外部文件复制工具。将外部文件复制到 workspace 目录，保留文件名。如果文件已存在会自动跳过。这是代码提取流程的第一步——将目标源文件纳入工作区管理。

- **参数**: `source_path`（外部绝对路径）

#### `read_function`

精确函数提取工具。使用 AST 解析从文件中提取指定函数或类的完整定义代码（含装饰器）。支持正则匹配和多函数同时提取。比 `read_file` + 手动寻找更高效，尤其对大文件。

- **参数**: `file_path`、`function_name`（支持正则，如 `_chunk_cat|_pad_chunk`）

#### `grep_search`

正则搜索工具。在指定路径下搜索匹配正则表达式的内容，返回匹配行及上下文。支持目录递归搜索，自动忽略二进制文件和常见非代码目录。搜索外部代码仓时不受 workspace 限制。

- **参数**: `pattern`（正则表达式）、`path`（搜索路径）、`max_results`（默认 20）

#### `save_to_workspace`

文本保存工具。将文本内容直接保存到 workspace 目录的指定文件名下。比 `write_file` 更简洁——不需要拼路径。

- **参数**: `filename`、`content`

#### `list_workspace`

工作区文件列表工具。列出当前 workspace 目录下的所有文件及其大小。

- **参数**: 无

#### `multi_file_search`

多文件代码片段搜索工具。同时在多个文件中搜索代码片段，返回每个文件中的匹配结果。适合在多个源文件中交叉定位函数调用关系。

- **参数**: `files`（文件路径列表）、`pattern`（搜索模式）

## 工具依赖关系

典型任务构建流程中，工具之间存在以下依赖关系：

```
Phase 1 (搜索定位)          Phase 2 (提取分析)              Phase 3 (构建)         Phase 4 (验证)
┌─────────────┐          ┌──────────────────┐          ┌──────────────┐      ┌───────────────┐
│ grep_search │────┐     │  read_function   │────┐     │ assemble_task│──────│ validate_task │
└─────────────┘    │     └──────────────────┘    │     └──────────────┘      └───────┬───────┘
                   ├────▶┌──────────────────┐    ├────▶                              │
                   │     │copy_to_workspace │    │                           ┌───────▼───────┐
                   │     └──────────────────┘    │                           │ optimize_task │
                   │                             │                           └───────┬───────┘
                   │     ┌──────────────────┐    │                                   │
                   └────▶│trace_dependencies│────┘                           ┌───────▼───────┐
                         └──────────────────┘                                │ validate_task │
                                                                             └───────┬───────┘
                                                                                     │
                                                                         ┌───────────▼──────────┐
                                                                         │ test_with_reference  │
                                                                         └──────────────────────┘
```

**可并行的组合**:
- 多个 `grep_search`（搜索不同关键词）
- 多个 `copy_to_workspace`（复制不同文件）
- 多个 `read_function`（从不同文件提取函数）
- `write_file`（reference_code）+ `write_file`（multi_inputs_code）

**必须串行的链路**:
- `grep_search` → `copy_to_workspace` → `read_function` → `trace_dependencies`
- `assemble_task` → `validate_task` → `optimize_task` → `validate_task`
- `test_with_reference` → `finish`

## 消息管理机制

TaskConstructor 使用完整的 chat completion 历史，每步都将 `self.messages` 传给 LLM。为控制上下文长度，实现了 `_manage_history()` 压缩机制：

- **触发条件**: 消息数超过 `MAX_MESSAGES=50`
- **保留内容**: system prompt + 初始用户消息 + 压缩摘要 + 最近的消息
- **压缩策略**: 中间消息被提取为 `[操作历史摘要]`，保留关键信息（action + thought 摘要、workspace 文件引用）
- **效果**: 实测 37 步任务中，最大 prompt 约 69KB，经压缩后回落到 64KB

## 使用方式

### 方式一：直接调用（测试用）

```python
import asyncio
from akg_agents.op.agents.task_constructor import TaskConstructor

async def main():
    agent = TaskConstructor()
    result = await agent.run(
        user_input="提取 torch._chunk_cat 的分解实现，构造标准化任务",
        source_path="/path/to/pytorch",
    )
    print(result["status"])
    print(result["task_code_path"])

asyncio.run(main())
```

### 方式二：通过 KernelAgent 主流程调用

KernelAgent 在启动时自动注册 `call_task_constructor` 工具。LLM 根据 Skill 指南自动选择调用：

```python
import asyncio
from akg_agents.op.agents.kernel_agent import KernelAgent

async def main():
    # 注意: KernelAgent.run() 只接受 user_input，
    # source_path 应包含在描述中，由 LLM 在调用工具时提取
    agent = KernelAgent(task_id="test_001")
    result = await agent.run(
        user_input="在路径 /path/to/pytorch 里找到 torch._chunk_cat，生成 triton 算子",
    )

asyncio.run(main())
```

KernelAgent → LLM 决策 → `call_task_constructor(user_input=..., source_path=...)` → ToolExecutor → `TaskConstructor.run()`

## Skill 系统

### Skill 文件

位于 `op/resources/skills/task-constructor/SKILL.md`

### 资源文件

| 文件 | 说明 | 调用方式 |
|------|------|----------|
| `references/kernelbench-format.md` | KernelBench 格式规范 | `read_file` 读取 |
| `references/assembly-strategies.md` | 装配策略说明 | `read_file` 读取 |
| `scripts/validate_kernelbench_task.py` | 格式验证脚本 | `execute_script` 执行 |

资源文件不会被 SkillRegistry 自动加载，而是在 SKILL.md 中作为文档引用，由 LLM 按需通过工具读取/执行。

## 日志系统

每次运行在 `~/.akg/task_constructor/<timestamp>/logs/` 下生成：

| 文件 | 内容 |
|------|------|
| `system_prompt.txt` | 完整 system prompt |
| `initial_message.txt` | 初始用户消息 |
| `session.log` | 文本格式会话日志（含每步 LLM 响应摘要） |
| `messages.jsonl` | 结构化步骤记录（action/thought/result） |
| `prompt_final.json` | 最后一步发送给 LLM 的完整 messages 列表 |
| `result.json` | 最终结果 |
| `task_output.py` | 生成的任务代码 |

## 目录结构

```
python/akg_agents/op/
├── agents/
│   └── task_constructor.py          # Agent 主实现（ReAct 循环、SessionLogger）
├── tools/task_constructor/
│   ├── __init__.py                  # 入口，触发工具注册
│   ├── tool_registry.py             # TaskToolRegistry（工具注册/执行/列举）
│   ├── code_tools.py                # 代码工具注册入口（注册 assembly + execution 工具）
│   ├── file_tools.py                # 文件操作工具（read/write/scan/grep/copy 等）
│   ├── path_utils.py                # 共享路径解析（workspace/output 前缀处理）
│   ├── ast_utils.py                 # AST 解析（函数提取、依赖追踪、import 别名解析）
│   ├── code_cleanup.py              # 代码清理（import 去重、私有模块移除、格式化）
│   ├── assembly.py                  # 任务装配、依赖追踪、验证、优化
│   └── execution.py                 # 代码执行、对比测试、补丁应用
└── resources/
    ├── prompts/task_constructor/
    │   └── system_prompt.j2         # Jinja2 prompt 模板（工具描述注入）
    └── skills/task-constructor/
        ├── SKILL.md                 # Skill 描述（供 KernelAgent 注入 prompt）
        ├── references/              # 参考文档（由 LLM 通过 read_file 读取）
        └── scripts/                 # 验证脚本（由 LLM 通过 execute_script 执行）
```

## 常见问题

**Q: 为什么某些任务需要 30+ 步？**  
A: 复杂算子（如 `_chunk_cat`）需要追踪多级依赖、内联外部函数、修复签名不兼容问题。典型简单任务可在 15-20 步内完成。

**Q: validate_task 失败但代码看起来正确？**  
A: validate_task 在隔离 subprocess 中执行，环境可能缺少某些模块。用 `run_code` 手动测试可以获得更详细的错误信息。

**Q: 如何查看完整的 LLM prompt？**  
A: 运行结束后查看 `~/.akg/task_constructor/<timestamp>/logs/prompt_final.json`，包含最后一步发送给 LLM 的完整 messages 列表。`system_prompt.txt` 和 `initial_message.txt` 分别包含系统 prompt 和初始消息。

**Q: 从 KernelAgent 调用时如何传 source_path？**  
A: 将路径包含在 `user_input` 描述中，例如 "在路径 /path/to/repo 里找到..."。KernelAgent 的 LLM 会在调用 `call_task_constructor` 工具时自动提取 `source_path` 参数。
