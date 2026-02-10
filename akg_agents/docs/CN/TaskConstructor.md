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

## 工具集（17 个）

### 代码分析

| 工具 | 功能 |
|------|------|
| `trace_dependencies` | AST 依赖追踪，发现所有被调用函数 |
| `assemble_task` | 从源文件提取函数，组装自包含任务文件 |
| `validate_task` | 预运行验证（实例化 → forward → NaN/Inf） |
| `optimize_task` | 清理 import、移除无用代码、格式化 |
| `test_with_reference` | 与原始函数多组输入对比 |

### 代码执行

| 工具 | 功能 |
|------|------|
| `run_code` | 运行 Python 代码或文件 |
| `apply_patch` | 通过字符串替换修改文件 |

### 文件操作

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件（支持行范围） |
| `write_file` | 写入文件 |
| `append_to_file` | 追加内容 |
| `scan_dir` | 扫描目录结构 |
| `copy_to_workspace` | 复制外部文件到工作区 |
| `read_function` | 精确提取函数/类代码 |
| `grep_search` | 正则搜索文件内容 |
| `save_to_workspace` | 保存文本到工作区文件 |
| `list_workspace` | 列出工作区文件 |
| `multi_file_search` | 多文件代码片段搜索 |

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

KernelAgent 在启动时自动注册 `call_task_constructor` 工具。在对话中，LLM 会根据 Skill 指南自动选择调用：

```python
from akg_agents.op.agents.kernel_agent import KernelAgent

agent = KernelAgent(task_id="test_001")
result = await agent.run(
    user_input="从 pytorch 仓中提取 torch._chunk_cat 的实现，生成 triton 算子",
    source_path="/path/to/pytorch",
)
```

KernelAgent 会通过 `ToolExecutor` 自动调用 `call_task_constructor`，参数从 LLM 输出中提取。

## Skill 系统

### Skill 文件

位于 `op/resources/skills/task-constructor/SKILL.md`

### 资源文件

| 文件 | 说明 |
|------|------|
| `references/kernelbench-format.md` | KernelBench 格式规范 |
| `references/assembly-strategies.md` | 装配策略说明 |
| `scripts/validate_kernelbench_task.py` | 格式验证脚本 |

### 资源加载方式

- **references**：由 LLM 通过 `read_file` 工具读取
- **scripts**：由 LLM 通过 `execute_script` 工具执行

```
# 读取参考文档
Action: read_file(file_path="resources/skills/task-constructor/references/kernelbench-format.md")

# 执行验证脚本
Action: execute_script(
    script_path="resources/skills/task-constructor/scripts/validate_kernelbench_task.py",
    args="--stdin --json",
    stdin_input="<task 代码>"
)
```

## 日志系统

每次运行在 `~/.akg/task_constructor/<timestamp>/logs/` 下生成：

| 文件 | 内容 |
|------|------|
| `system_prompt.txt` | 完整 system prompt |
| `initial_message.txt` | 初始用户消息 |
| `session.log` | 文本格式会话日志 |
| `messages.jsonl` | 结构化步骤记录 |
| `prompt_step_NNN.json` | 每步发送给 LLM 的完整 messages |
| `result.json` | 最终结果 |
| `task_output.py` | 生成的任务代码 |

## 目录结构

```
python/akg_agents/op/
├── agents/
│   └── task_constructor.py          # Agent 主实现
├── tools/task_constructor/
│   ├── __init__.py                  # 入口，触发工具注册
│   ├── tool_registry.py             # TaskToolRegistry
│   ├── code_tools.py                # 代码工具注册入口
│   ├── file_tools.py                # 文件操作工具
│   ├── path_utils.py                # 共享路径解析
│   ├── ast_utils.py                 # AST 解析
│   ├── code_cleanup.py              # 代码清理
│   ├── assembly.py                  # 任务装配/验证
│   └── execution.py                 # 代码执行/测试
└── resources/
    ├── prompts/task_constructor/
    │   └── system_prompt.j2         # Jinja2 prompt 模板
    └── skills/task-constructor/
        ├── SKILL.md                 # Skill 描述
        ├── references/              # 参考文档
        └── scripts/                 # 验证脚本
```
