# AKG CLI

AKG CLI（`akg_cli`）是 AI Kernel Generator 的命令行入口。本文聚焦两个最常用流程：

- 启动 Worker Service：`akg_cli worker --start`
- 生成算子：`akg_cli op ...`

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
- `--worker-url` 支持多个 Worker 地址，使用逗号分隔。
- `--devices` 将本地设备注册到本地 server（无需 worker_url）。与 `--worker-url` 互斥，且仅支持本地 server。
- `--output-path` 指定 `saved_verifications` 的保存根目录（默认：启动 `akg_cli` 时的当前工作目录）。
- `--stream/--no-stream` 控制是否启用流式输出。
- `--yes` 自动确认所有提示。
- `--ipv6` 强制本地 server 使用 IPv6（仅自动拉起本地 server 时生效）。

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

IPv6-only 示例（注意方括号）：

```bash
akg_cli worker --start --host :: --port 9001
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url [::1]:9001
```

## 3. TUI 基本说明

`akg_cli op` 默认使用 TUI 交互界面，主要包含以下区域：

- 顶部：任务 Tabs。
  - `main` 为主对话。
  - 子 Agent 在独立 Tab 中执行，需要切换查看其输出。
- 左侧：聊天日志（主输出区域）。
- 右侧：任务信息面板、流程面板、Trace 列表。
- 底部：输入框。

焦点与导航：

- `Tab` 在各面板间切换焦点。
- 在 Trace 列表中选择条目，可跳转到左侧日志的对应位置。
- `F8` 切换到下一个并发任务（出现任务 Tabs 时）。

常用快捷键：

- `Ctrl+C`：退出。
- `Ctrl+E`：滚动到底部。
- `Enter`：提交输入（在输入框内）。
- `Ctrl+J`：插入换行（在输入框内）。

说明：
- 生成期间输入被禁用时，输入框占位符会提示 `Ctrl+C` 可取消当前生成。
