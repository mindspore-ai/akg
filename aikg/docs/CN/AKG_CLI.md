# AKG CLI

AKG CLI（`akg_cli`）是 AI Kernel Generator 的命令行入口。本文聚焦两个最常用流程：

- 启动 Worker Service：`akg_cli worker --start`
- 生成算子：`akg_cli op ...`

二次开发者（想扩展消息类型/自定义面板/理解 MainOpAgent 与 TUI 的交互）建议阅读《[AKG_CLI 二次开发指南](./AKG_CLI_Develop.md)》。

查看帮助：

```bash
akg_cli --help
akg_cli op --help
akg_cli worker --help
```

## 0. 安装

首先，在 `aikg` 目录下执行以下命令安装 `akg_cli`：

```bash
pip install -e .
```

## 1. 启动 Worker Service

Worker Service 提供算子生成的后端运行时，执行 `akg_cli op` 前需要先启动。

```bash
# Ascend 910B2
akg_cli worker --start --backend ascend --arch ascend910b2 --devices 0 --host 127.0.0.1 --port 9001

# CUDA A100: --backend cuda --arch a100
# akg_cli worker --start --backend cuda --arch a100 --devices 0 --host 127.0.0.1 --port 9001
```

说明：

- `--backend` 支持 `cuda` 或 `ascend`。未指定时优先读取 `WORKER_BACKEND`，否则默认 `cuda`。
- `--arch` 未指定时优先读取 `WORKER_ARCH`，否则默认 `a100`。
- `--devices` 为本地设备列表，逗号分隔（如 `0` 或 `0,1,2`），不能为空、不能重复、不能包含负数。
- `--host` 默认 `0.0.0.0`，`--port` 默认 `9001`。IPv6-only 机器可使用 `--host ::`（双栈/IPv6 监听）。

需要停止服务时：

```bash
akg_cli worker --stop --port 9001
```

## 2. 生成算子

使用 `akg_cli op` 启动算子生成。目标配置必须通过命令行显式提供。

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001

# CUDA A100: --backend cuda --arch a100 --dsl triton_cuda
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001
```

可选参数：

- `--intent "..."` 直接输入需求（跳过交互提示）。
- `--task-file <path>` 直接读取 task_desc 文件（KernelBench 格式），跳过 OpTaskBuilder 的转换流程。适用于已有完整 task_desc 代码的场景。
- `--worker-url` 支持多个 Worker 地址，使用逗号分隔。CLI 同时接受 `--worker-url` 与 `--worker_url` 两种写法。
- 下面两种方式二选一：
  - 使用远端 Worker Service 时用 `--worker-url/--worker_url`。
  - 使用本地设备时用 `--devices`。
  - `--devices` 与 `--worker-url/--worker_url` 互斥，不能同时指定。
- `--output-path` 指定 `saved_verifications` 的保存根目录（默认：启动 `akg_cli` 时的当前工作目录）。
- `--stream/--no-stream` 控制是否启用流式输出（默认启用 `--stream`）。
- `--rag/--no-rag` 控制是否启用 RAG（默认关闭 `--no-rag`）。
- `--yes/-y` 自动确认所有提示。

提示：如果希望“尽量非交互”，可使用 `--intent "..." --yes`（尽力而为；在部分场景下 CLI 仍可能需要你补充输入）。

### ReAct 模式注意事项

- `ask_user` **不会再使用 `input()` 抢占 stdin**：已改为基于 LangGraph `interrupt` 的可恢复暂停机制。
  - 当 Agent 需要你确认/补充信息时，会触发一次中断并展示问题；
  - 你在下一轮输入的内容会作为 `Command(resume=...)` 回填给 `ask_user`，Agent 随后继续执行。
- 如果你依赖 openai-compatible endpoint（例如 vLLM，或通过 `AIKG_BASE_URL/AIKG_MODEL_NAME/AIKG_API_KEY` 指定模型），react 模式需要安装 `langchain-openai`：

```bash
pip install -U langchain-openai
```

示例：

```bash
# Ascend 910B2 带 intent
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --intent "实现 fused softmax，输入为 [batch, head, seq, dim]"

# CUDA A100 带 intent
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001 --intent "实现 fused softmax，输入为 [batch, head, seq, dim]"
```

多 Worker 示例：

```bash
# Ascend 910B2 使用多个 Worker
akg_cli op --framework mindspore --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001,127.0.0.1:9002
```

本地设备示例（无需 worker URL）：

```bash
# Ascend 910B2 使用本地设备
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --devices 0

# CUDA A100 使用本地设备
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --devices 0
```

使用 task_file 示例（跳过 OpTaskBuilder 转换）：

```bash
# 直接读取已有的 task_desc 文件，跳过 OpTaskBuilder 转换
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --task-file ./my_task_desc.py --intent "生成优化的 kernel"
```

说明：
- `--task-file` 适用于已有完整 KernelBench 格式 task_desc 代码的场景
- 文件应包含 `class Model`、`get_inputs()`、`get_init_inputs()` 等标准组件
- 使用该参数后，CLI 会跳过 OpTaskBuilder 的自然语言理解和代码转换步骤，直接使用文件内容进行算子生成

IPv6-only 示例（注意方括号）：

```bash
akg_cli worker --start --host :: --port 9001
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url [::1]:9001
```

## 3. 交互式 CLI 基本说明

`akg_cli op` 会进入交互式输入循环；你可以在同一会话中进行多轮输入与生成。

输入：

- `Enter`：提交当前输入。
- `Ctrl+J`：插入换行（支持多行输入）。

退出/取消：

- 空闲时按 `Ctrl+C` 退出。
- 生成中按 `Ctrl+C` 取消当前这一轮生成；随后可继续下一轮输入，或再次按 `Ctrl+C` 退出。

信息面板：

- `F2`：显示/隐藏信息面板。
- 面板用于展示当前任务状态、输出/保存路径，以及最近 Top 实现摘要。

历史输入：

- 使用上下方向键浏览历史输入。
- 历史记录保存于 `~/.akg_cli_history`。
