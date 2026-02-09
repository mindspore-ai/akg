# 算子任务构建 Demo

基于 ReAct（Reasoning + Acting）模式的算子任务自动构建工具。将用户提供的 PyTorch/Triton 代码自动转化为 KernelBench 标准格式的单文件自包含任务。

## 快速开始

```bash
# 交互模式（推荐）
python main.py

# 指定文件 + 描述
python main.py -i path/to/code.py -d "RMS归一化算子"

# 指定目录
python main.py -i path/to/repo/ -d "提取mmdit算子"

# 选择 LLM 模型级别
python main.py -i code.py -m complex
```

### 交互模式示例

```
输入 > C:\Users\me\my_ops\my_kernel

描述 > 帮我构造 my_kernel_func 这个函数的任务

开始处理...
Step 1/50 思考: 扫描目录...
Step 2/50 思考: 复制文件到工作区...
...
Step 24/50 思考: 验证通过...
结果: success (24 步)
```

## 目录结构

```
demo/
├── main.py                     # 入口: CLI 解析、交互模式、结果输出
├── config.py                   # 全局配置: 路径、LLM 客户端、超时等
├── agent/
│   ├── react_loop.py           # ★ ReAct 主循环: LLM调度、工具执行、消息管理
│   └── prompts.py              # System Prompt 模板（规则、策略、格式）
├── tools/
│   ├── registry.py             # ToolRegistry: 工具注册中心
│   ├── file_tools.py           # 文件工具: read_file, write_file, scan_dir, grep等
│   ├── code_tools.py           # 代码工具: assemble_task, apply_patch, run_code等
│   └── user_tools.py           # 交互工具: ask_user
├── task/
│   ├── input_parser.py         # 输入解析: 代码/文件/目录识别
│   ├── task_builder.py         # KernelBench 格式模板与格式验证
│   └── test_constructor.py     # 预运行验证: 实例化+forward+NaN/Inf+一致性检查
└── output/                     # 运行时输出目录
    ├── workspace/              # 工作区（每次运行前自动清理）
    ├── logs/                   # 每次运行的完整日志
    │   └── YYYYMMDD_HHMMSS/
    │       ├── session.log     # 人类可读的完整过程日志
    │       ├── messages.jsonl  # 结构化步骤记录
    │       ├── system_prompt.txt
    │       ├── initial_message.txt
    │       ├── result.json
    │       └── task_output.py  # 最终生成的任务文件副本
    └── task_output.py          # 最终输出的任务文件
```

## 核心架构

### 1. ReAct 循环 (`agent/react_loop.py`)

ReAct = Reasoning（思考）+ Acting（行动）。每一步 LLM 输出 JSON：

```json
{
    "thought": "我需要分析目标函数的依赖链...",
    "action": "read_function",
    "arguments": {"file_path": "workspace/src.py", "function_name": "my_func"}
}
```

**完整流程**：

```
用户输入 → InputParser.parse()
         → ReactAgent.run()
           ├── _clean_workspace()           # 清理上次工作区
           ├── _build_initial_message()     # 构建初始提示
           └── for step in 1..50:           # ReAct 循环
               ├── _get_next_action()       # 调用 LLM
               │   ├── LLM.generate()       # 异步调用
               │   ├── _parse_action()      # 解析 JSON（多策略容错）
               │   └── 重试 (最多3次)
               ├── if "finish" → 结束       # 格式检查 + 返回
               ├── ToolRegistry.execute()   # 执行工具
               ├── _build_tool_result_msg() # 格式化结果
               ├── _manage_history()        # 消息管理（防溢出）
               └── SessionLogger.log_*()    # 记录日志
```

**关键机制**：
- **消息管理** (`_manage_history`): 保留最近 50 条消息，移除时保留 workspace 相关引用
- **JSON 解析容错**: 支持 markdown code block 包裹、截断检测、多种格式修复
- **workspace 路径解析**: `workspace/xxx.py` 自动映射到 `output/workspace/`

### 2. 工具系统 (`tools/`)

#### 工具注册模式

所有工具通过 `ToolRegistry.register()` 统一注册：

```python
ToolRegistry.register(
    "tool_name",                          # 工具ID
    "工具描述（给LLM看的自然语言说明）",    # description
    {                                      # JSON Schema 参数定义
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数说明"},
        },
        "required": ["param1"],
    },
    tool_function,                         # 执行函数: (args: dict) -> dict
)
```

执行函数的返回格式统一为：
```python
{"status": "success"|"error", "output": "结果文本", "error": "错误信息"}
```

#### 工具清单

| 工具 | 模块 | 功能 |
|------|------|------|
| `scan_dir` | file_tools | 浏览目录结构（文件名+行数+函数概览） |
| `read_file` | file_tools | 读取文件内容（支持 offset/limit） |
| `write_file` | file_tools | 写入文件（相对路径→output/） |
| `append_to_file` | file_tools | 追加内容到文件末尾（分段生成用） |
| `copy_to_workspace` | file_tools | 复制整个源文件到工作区 |
| `save_to_workspace` | file_tools | 手动保存内容到工作区 |
| `list_workspace` | file_tools | 列出工作区所有文件 |
| `read_function` | file_tools | AST 精确提取函数/类，自动保存到工作区 |
| `grep_search` | file_tools | 在文件/目录中搜索文本 |
| `apply_patch` | code_tools | 查找替换文本（精确修改代码） |
| `run_code` | code_tools | 运行 Python 代码片段或文件 |
| `assemble_task` | code_tools | ★核心：拼装 KernelBench 任务文件 |
| `validate_task` | code_tools | 预运行验证（实例化+forward+检查） |
| `ask_user` | user_tools | 向用户提问获取确认 |

#### assemble_task 详解

三种模式：

```python
# 模式1: 完整嵌入
source_files=["workspace/file.py"]

# 模式2: 选择性提取（只提取指定函数，自动包含 import）
source_files=[{"path": "workspace/file.py", "functions": ["func1", "func2"]}]

# 模式3: 排除模式（嵌入整个文件但移除指定函数）
source_files=[{"path": "workspace/file.py", "exclude_functions": ["unused_func"]}]
```

其他参数：
- `model_code`: Model 类代码（LLM 编写）
- `helper_code`: 辅助代码（常量、配置函数等）
- `get_inputs_code`: `get_inputs()` 函数
- `get_init_inputs_code`: `get_init_inputs()` 函数
- `output_file`: 输出文件名（默认 `task_output.py`）

### 3. 任务格式 (`task/`)

#### KernelBench 标准格式

```python
import torch
import torch.nn as nn

# ... 所有依赖函数（内联，不能 import 外部文件）...

class Model(nn.Module):
    def __init__(self, <config_params>):
        super(Model, self).__init__()
        ...

    def forward(self, <input_tensors>):
        return output  # 单个 Tensor 或 Tuple[Tensor, ...]

def get_inputs():
    return [input1, input2, ...]  # 与 forward 参数顺序一致

def get_init_inputs():
    return [param1, ...]  # 与 __init__ 参数顺序一致
```

**关键要求**：
- 文件必须**自包含**，所有函数内联
- `get_inputs()` 返回列表，可包含 Tensor 和标量
- `forward` 可返回单个 Tensor 或多个 Tensor 的 tuple

#### 预运行验证 (`test_constructor.py`)

验证流程：
1. `Model(*get_init_inputs())` — 实例化成功
2. `get_inputs()` — 生成输入成功，打印 shape/dtype
3. `model(*inputs)` — forward 执行成功
4. 检查输出无 NaN/Inf
5. 一致性检查：两次运行结果完全一致
6. 输出统计：mean/std/min/max（辅助人工判断）

### 4. 配置 (`config.py`)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_REACT_STEPS` | 50 | 最大 ReAct 步数 |
| `MAX_RETRIES_PER_STEP` | 3 | 每步 LLM 重试次数 |
| `CODE_EXEC_TIMEOUT` | 60s | 代码执行超时 |
| `READ_FILE_MAX_LINES` | 300 | 单次读取最大行数 |
| `OUTPUT_DIR` | `demo/output/` | 输出目录 |
| `WORKSPACE_DIR` | `demo/output/workspace/` | 工作区目录 |

LLM 客户端通过 `get_llm_client()` 复用 `akg_agents` 已有的工厂函数。

### 5. 日志系统 (`SessionLogger`)

每次运行自动创建 `output/logs/YYYYMMDD_HHMMSS/` 目录：

- `session.log`: 人类可读的完整过程（思考、动作、结果、LLM原始响应）
- `messages.jsonl`: 结构化记录（每步一行JSON）
- `system_prompt.txt`: 本次使用的完整 system prompt
- `initial_message.txt`: 发送给 LLM 的初始消息
- `result.json`: 最终结果（状态、摘要、步数等）
- `task_output.py`: 生成的任务文件副本

## Agent 工作流程详解

Agent 遵循以下标准工作流（由 `prompts.py` 引导）：

### 第一步：分析结构（1-4步）
1. `scan_dir` 浏览用户提供的目录
2. `copy_to_workspace` 复制关键文件（源文件 + benchmark/test）

### 第二步：依赖分析（5-15步）
3. `read_function` 提取目标函数
4. 分析函数调用链，列出直接/间接依赖
5. `read_function` 提取 benchmark 的 `create_config`/`create_inputs`
6. 检查 NPU/Triton 特定代码是否有 fallback

### 第三步：构建任务（1-3步）
7. 选择策略：
   - **排除模式**（首选）：依赖大部分函数时，排除不需要的
   - **选择性提取**：只依赖少数函数时
   - **分段追加**：需要修改源函数时
8. 调用 `assemble_task` 或 `write_file` + `append_to_file`

### 第四步：验证（1-3步）
9. `validate_task` 预运行验证
10. 如有错误，用 `apply_patch` 修复
11. `finish` 完成

### 重要规则

- **禁止重写复杂函数**：原始代码的索引计算很精确，自己写的简化版几乎必定有 bug
- **返回值必须完全一致**：不能截断返回值
- **不确定时 ask_user**：不要自行假设
- **每次 JSON 中代码不超过 150 行**：避免 LLM 输出被截断

## 工作区机制

### 生命周期

```
ReactAgent.run() 开始
  → _clean_workspace()     # 删除并重建 workspace 目录
  → copy_to_workspace()    # 复制源文件
  → read_function()        # 自动保存提取的函数
  → assemble_task()        # 从 workspace 读取文件生成任务
  → validate_task()        # 验证
ReactAgent.run() 结束
  → workspace 保留（下次运行前清理）
```

- **每次运行前自动清理**：不会混入上次的残留
- **工作区文件命名**：`{源文件名}__{函数名}.py`
- **路径前缀**：所有工具支持 `workspace/` 前缀快速引用

### import 过滤机制

选择性提取时，工具使用 `importlib.util.find_spec()` 检测模块：
- 标准库/已安装包 → 保留
- 本地文件模块（如 `from my_module import ...`）→ 自动移除

## 依赖

- Python 3.9+
- `torch`（用于验证）
- `akg_agents` 的 LLM 客户端（需确保 `settings.json` 或环境变量配置正确）
