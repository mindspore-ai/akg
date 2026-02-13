[English Version](../AKG_CLI.md)

# AKG CLI

## 1. 概述

`akg_cli` 是 AKG Agents 的命令行工具，提供对框架所有能力的交互式访问。

> 安装与基础配置请参考 [README](../../README_CN.md#️-3-快速上手)。

## 2. 命令总览

| 命令 | 说明 |
|------|------|
| `akg_cli op` | Kernel Agent — 多后端、多 DSL 算子生成。详见 [Kernel Agent](./KernelAgent.md)。 |
| `akg_cli common` | Common Agent — 通用 ReAct Agent（演示）。 |
| `akg_cli worker --start` | 启动 Worker Service，用于分布式执行。 |
| `akg_cli worker --stop` | 停止 Worker Service。 |
| `akg_cli sessions` | 列出所有带 trace 历史的会话。 |
| `akg_cli resume <session_id>` | 恢复之前的会话。 |
| `akg_cli list` | 列出支持的子命令。 |

## 3. akg_cli op

AI 算子代码生成的主命令。

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--framework` | 计算框架 | `torch`、`mindspore` |
| `--backend` | 硬件后端 | `ascend`、`cuda`、`cpu` |
| `--arch` | 硬件架构 | `ascend910b2`、`ascend910b4`、`a100`、`x86_64` |
| `--dsl` | 目标 DSL | `triton_ascend`、`triton_cuda`、`cuda`、`tilelang_cuda`、`cpp` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--intent` | `None` | 直接提供需求文本（跳过交互式输入） |
| `--task-file` | `None` | 读取任务描述文件（KernelBench 格式） |
| `--devices` | `None` | 本地设备列表，逗号分隔（如 `0,1,2,3`） |
| `--worker-url` | `None` | Worker Service 地址，逗号分隔 |
| `--stream/--no-stream` | `--stream` | 启用/关闭 LLM 流式输出 |
| `--rag/--no-rag` | `--no-rag` | 启用/关闭 RAG 检索 |
| `--resume` | `None` | 通过 ID 恢复之前的会话 |
| `-y` | `False` | 自动确认所有提示 |

### 示例

```bash
# Ascend 910B2 + Triton
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100 + Triton
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0,1,2,3,4,5,6,7

# 直接指定需求
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0 --intent "帮我生成一个 relu 算子"

# 使用 KernelBench 任务文件
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0 --task-file path/to/task.py
```

## 4. Worker Service

用于跨多台机器的分布式执行：

```bash
# 在每台机器上启动 worker
akg_cli worker --start --port 9001

# 从客户端连接
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --worker-url machine1:9001,machine2:9002
```

## 5. 会话管理

```bash
# 列出所有会话
akg_cli sessions list

# 恢复会话
akg_cli resume <session_id>
```
