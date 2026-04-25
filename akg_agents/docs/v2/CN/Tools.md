[English Version](../Tools.md)

# Tools 体系

## 1. 概述

Tools 模块提供了 AKG Agents 的工具执行框架。Agent 通过工具与外部环境交互 —— 读写文件、运行脚本、验证算子等。

核心组件：

- **ToolExecutor** — 执行工具、解析参数、捕获错误、持久化结果
- **Basic Tools** — 内置通用工具（文件读写、Shell、代码检查）
- **Domain Tools** — 场景专用工具（算子验证、性能分析）
- **工具配置** — 基于 YAML 的工具定义（`tools.yaml`、`domain_tools.yaml`）

## 2. ToolExecutor

`ToolExecutor` 是 `ReActAgent` 使用的核心工具执行引擎，它：

1. 接收 LLM 输出的工具名称和参数
2. 解析参数表达式（如 `read_json_file('path')['key']`）
3. 分发到对应的工具函数
4. 捕获执行期间的错误（WARNING+ 级别日志、stderr）
5. 将结果持久化到 trace 节点路径

### 核心特性

- **参数解析**：通过 `arg_resolver` 支持工具参数中的表达式，如 `read_json_file('result.json')['output']`
- **错误捕获**：捕获工具执行期间的 WARNING+ 级别日志和 stderr，将精简错误摘要注入结果
- **Agent/Workflow 分发**：除常规工具外，还可分发到已注册的 Agent（`call_agent` 类型）或 Workflow（`call_workflow` 类型）

## 3. 内置工具（basic_tools）

| 工具 | 说明 | 必需参数 |
|------|------|----------|
| `read_file` | 读取文件内容。支持相对路径和绝对路径。 | `file_path` |
| `write_file` | 将内容写入文件。支持按 op_name 自动命名。 | `content` |
| `ask_user` | 向用户提问并等待回复。 | `message` |
| `check_python_code` | 检查 Python 语法并使用 autopep8 自动格式化。 | `file_path` |
| `check_markdown` | 使用 markdownlint-cli 检查 Markdown 格式。 | `file_path` |
| `execute_script` | 执行 Shell 或 Python 脚本。 | `script_path` |

### write_file 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `content` | string | 是 | 要写入的文件内容 |
| `file_path` | string | 否 | 显式文件路径。未提供时根据 `op_name` 和 `file_type` 自动生成 |
| `op_name` | string | 否 | 算子名称，用于自动命名 |
| `file_type` | string | 否 | 文件类型/扩展名，用于自动命名 |
| `encoding` | string | 否 | 文件编码（默认 `"utf-8"`） |
| `overwrite` | bool | 否 | 是否覆盖已有文件（默认 `True`） |

### execute_script 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `script_path` | string | 是 | 要执行的脚本路径 |
| `args` | list | 否 | 传递给脚本的命令行参数 |
| `stdin_input` | string | 否 | 传递给脚本的标准输入 |
| `timeout` | int | 否 | 执行超时时间，秒（默认 `300`） |
| `working_dir` | string | 否 | 脚本执行的工作目录 |

### 返回格式

所有内置工具返回标准化字典：

```python
{
    "status": "success" | "error",
    "output": "...",           # 工具输出
    "error_information": "..."  # 错误详情（成功时为空）
}
```

## 4. 领域工具（domain_tools）

领域工具是面向算子生成与优化场景的专用工具。

### verify_kernel

通过对比框架实现（如 PyTorch）和生成实现（如 Triton）的输出结果，验证生成算子代码的正确性。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `task_code` | string | 是 | 框架实现代码（需包含 `class Model`） |
| `generated_code` | string | 是 | 待验证的生成代码（需包含 `class ModelNew`） |
| `op_name` | string | 是 | 算子名称（如 `"relu"`、`"matmul"`） |
| `task_id` | string | 否 | 任务 ID（默认 `"default_task"`） |
| `device_id` | int | 否 | 设备 ID（默认 `0`） |
| `timeout` | int | 否 | 超时时间，秒（默认 `300`） |
| `cur_path` | string | 否 | 当前 trace 节点路径（默认 `""`） |
| `framework` | string | 否 | 计算框架（默认 `"torch"`） |
| `backend` | string | 否 | 硬件后端（默认 `"cuda"`） |
| `arch` | string | 否 | 硬件架构（默认 `"a100"`） |
| `dsl` | string | 否 | 目标 DSL（默认 `"triton"`） |

### profile_kernel

算子代码性能分析，返回执行时间和加速比。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `task_code` | string | 是 | 框架实现代码 |
| `generated_code` | string | 是 | 待分析的生成代码 |
| `op_name` | string | 是 | 算子名称 |
| `task_id` | string | 否 | 任务 ID（默认 `"default_task"`） |
| `device_id` | int | 否 | 设备 ID（默认 `0`） |
| `run_times` | int | 否 | 性能测试运行次数（默认 `50`） |
| `warmup_times` | int | 否 | 预热次数（默认 `5`） |
| `cur_path` | string | 否 | 当前 trace 节点路径（默认 `""`） |
| `framework` | string | 否 | 计算框架（默认 `"torch"`） |
| `backend` | string | 否 | 硬件后端（默认 `"cuda"`） |
| `arch` | string | 否 | 硬件架构（默认 `"a100"`） |
| `dsl` | string | 否 | 目标 DSL（默认 `"triton"`） |

> 注意：`profile_kernel` 仅做性能分析，不验证正确性。

## 5. 工具配置

工具通过 YAML 配置文件定义：

- `tools.yaml` — 基础工具定义
- `domain_tools.yaml` — 领域专用工具定义

### 配置格式

```yaml
tools:
  read_file:
    type: "basic_tool"
    function:
      name: "read_file"
      description: "读取文件内容。"
      parameters:
        type: "object"
        properties:
          file_path:
            type: "string"
            description: "文件路径"
        required: ["file_path"]

  verify_kernel:
    type: "domain_tool"
    function:
      name: "verify_kernel"
      description: "验证生成的算子代码正确性。"
      parameters:
        type: "object"
        properties:
          task_code:
            type: "string"
            description: "框架实现代码"
          generated_code:
            type: "string"
            description: "待验证的生成代码"
          op_name:
            type: "string"
            description: "算子名称"
        required: ["task_code", "generated_code", "op_name"]
```

### 工具类型

| 类型 | 说明 |
|------|------|
| `basic_tool` | 通用工具，分发到 `basic_tools` 模块 |
| `domain_tool` | 领域专用工具，分发到 `domain_tools` 模块 |
| `call_agent` | 分发到已注册的 Agent（通过 `AgentRegistry`） |
| `call_workflow` | 分发到已注册的 Workflow（通过 `WorkflowRegistry`） |

## 6. 参数解析器

`arg_resolver` 模块支持工具调用中的动态参数解析。当 LLM 生成的工具参数包含表达式时，会在执行前自动解析。

### 支持的表达式

```python
# 读取 JSON 文件并提取字段
read_json_file('path/to/result.json')['output']

# 嵌套访问
read_json_file('path/to/result.json')['data']['code']
```

这使得 Agent 可以链式传递工具结果 —— 例如读取上一个工具的 `result.json` 输出，作为下一个工具的输入。
