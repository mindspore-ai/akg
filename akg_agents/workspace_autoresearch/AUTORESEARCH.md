# AutoResearch

AutoResearch 用 Claude Code 或 OpenCode 驱动算子优化。用户提供 reference 和
seed kernel，状态机按以下流程完成基线、计划、修改、评测和取舍：

```text
BASELINE -> PLAN -> EDIT -> pipeline -> KEEP / DISCARD / FAIL
                                      -> REPLAN / DIAGNOSE / FINISH
```

日常使用只需要 `/autoresearch`。阶段切换和约束由 hook/plugin 管理，不要手工
修改 `.ar_state/state.json` 或 `.ar_state/plan.md`。

## 1. 快速开始

默认评测环境为 Linux。先进入工作区并加载包含 Python、硬件 SDK 和 DSL 依赖的
环境脚本：

```bash
cd <akg-repo>/akg_agents/workspace_autoresearch
source ~/env.sh
```

确认目标设备和 Python 依赖可用，例如 Ascend 环境：

```bash
npu-smi info
python -c "import torch; import torch_npu"
```

启动 Claude Code 或 OpenCode TUI：

```bash
claude
# 或
opencode
```

在 TUI 中创建任务：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

常用可选参数：

| 参数 | 说明 |
|---|---|
| `--max-rounds N` | 最大优化轮数。 |
| `--eval-timeout SEC` | 单个 shape 的评测预算。 |
| `--output-dir DIR` | task_dir 父目录，默认 `ar_tasks/`。 |
| `--worker-url HOST:PORT` | 使用已启动的远端 worker。 |
| `--no-code-checker` | 对当前任务关闭 CodeChecker。 |

恢复任务：

```text
/autoresearch --resume
/autoresearch --resume <task_dir>
```

查看进度：

```bash
python scripts/dashboard.py <task_dir> --watch
```

OpenCode headless 单任务使用外层 loop：

```bash
source ~/env.sh
python .opencode/run_loop.py \
  --ref workspace/<op>_ref.py \
  --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

## 2. 输入文件

### 2.1 Reference

Reference 文件至少暴露：

- `Model`
- `get_init_inputs()`
- `get_inputs()` 或 `get_input_groups()`

推荐命名：

```text
workspace/<op>_ref.py
```

与 reference 同名的 `.json`、`.pt`、`.npz` 输入文件会随任务复制和远端
打包。路径应保持在 reference 所在目录内。

### 2.2 Kernel

单文件 DSL 的 seed kernel 暴露 `ModelNew`：

```text
workspace/<op>_kernel.py
```

目录型 DSL 的 `--kernel` 指向项目目录；Python wrapper 放在相邻的
`kernel.py`。实际可编辑范围写入 `task.yaml: editable_files`，agent 只能修改
这些文件。

例如 CATLASS：

```text
workspace/<op>/
  reference.py
  kernel.py
  catlass_op/
    CMakeLists.txt
    kernel/
    include/
    src/
```

```text
/autoresearch --ref workspace/<op>/reference.py \
  --kernel workspace/<op>/catlass_op \
  --op-name <op> --devices <device-id>
```

`backend`、`framework`、`dsl` 和 skill 类型由 `config.yaml: defaults`
指定；硬件 arch 由设备探测或 worker 提供。

## 3. 远端 worker

开发机运行 agent，Linux 评测机运行 worker。远端机器需要：

- AKG checkout；
- 可在非交互 SSH shell 中执行的 `env.sh`；
- 目标 backend、DSL、SDK 和编译工具链。

开发机先配置 SSH alias，再在工作区 `config.yaml` 中登记：

```yaml
remote_worker:
  hosts:
    eval-host:
      repo_path: /path/to/akg
      env_script: /path/to/env.sh
      ssh_alias: eval-host
```

启动 worker 和本地 SSH tunnel：

```bash
source ~/env.sh
akg_cli worker --remote-host eval-host --start \
  --backend ascend --arch ascend910b3 --devices 0
```

检查状态：

```bash
akg_cli worker --remote-host eval-host --status
```

通过 tunnel 创建任务：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices 0 --worker-url 127.0.0.1:<port>
```

停止 worker 和 tunnel：

```bash
akg_cli worker --remote-host eval-host --stop
```

修改 worker 侧代码后需要同步 checkout 并重启 worker，已运行的 daemon 不会自动
加载新代码。

## 4. 批量运行

Batch 目录使用以下布局：

```text
<batch_dir>/
  refs/<op>_ref.py
  kernels/<op>_kernel.py
```

准备和预检：

```bash
python scripts/batch/prepare.py <batch_dir>
python scripts/batch/verify.py <batch_dir>
```

使用正式 KernelVerifier 做完整预检：

```bash
python scripts/batch/verify.py <batch_dir> --full --devices <device-id>
# 远端 worker：
python scripts/batch/verify.py <batch_dir> --full \
  --worker-url 127.0.0.1:<port>
```

运行、监控和汇总：

```bash
python -u scripts/batch/run.py <batch_dir> --devices <device-id>
python scripts/batch/monitor.py <batch_dir>
python scripts/batch/summarize.py <batch_dir>
```

远端运行时给 `run.py` 增加
`--worker-url 127.0.0.1:<port>`。常用筛选参数为 `--only`、`--limit` 和
`--retry-errored`。

## 5. Pipeline 与 trace

每个 EDIT 轮次只实现当前 ACTIVE plan item，然后由 agent 运行：

```bash
python scripts/engine/pipeline.py "<task_dir>"
```

`pipeline.py` 依次执行 quick check、eval、KEEP/DISCARD/FAIL 结算和阶段切换。
它完成后应继续遵循最后一条 `[AR Phase: ...]` 指引，不要自行推断下一阶段。

Ascend 任务需要性能时序证据时可启用 trace：

```bash
python scripts/engine/pipeline.py "<task_dir>" --trace
```

该轮会在
`.ar_state/akg_verify/<op>/Iteration<op>_Step<round>_verify/` 下保留每个
shape 的 profiling 产物，主要包括：

- `kernel_details.csv`、`op_statistic.csv`：逐 kernel/op 耗时，优先阅读；
- `trace_view.json`：完整时间线，需要时用 Perfetto 或
  `chrome://tracing` 查看。

后续 PLAN、REPLAN 或 DIAGNOSE guidance 会自动发现已有 trace 并提示对应路径，
因此无需把 trace 内容手工复制进 plan 或状态文件。CUDA 路径不使用该 msprof
trace。

## 6. 任务状态和产物

常用文件：

| 文件 | 用途 |
|---|---|
| `task.yaml` | 任务配置和 `editable_files`。 |
| `.ar_state/state.json` | 当前 phase 和进度；只读。 |
| `.ar_state/plan.md` | 当前计划；由 `create_plan.py` 管理。 |
| `.ar_state/history.jsonl` | 每轮结果和 decision。 |
| `.ar_state/report.md` | FINISH 报告。 |
| `.ar_state/akg_verify/` | verify、profile 和可选 trace 产物。 |

恢复和排查：

| 场景 | 操作 |
|---|---|
| 会话中断 | `/autoresearch --resume <task_dir>` |
| 查看当前阶段 | `python scripts/dashboard.py <task_dir>` |
| worker 不可达 | 先运行 `akg_cli worker --remote-host <alias> --status` |
| tunnel 中断 | 再次执行同一条 worker `--start` 命令 |
| 正确性失败 | 查看 pipeline 摘要和对应 verify 目录 |
| 连续失败 | 让状态机进入 DIAGNOSE，不手改状态 |

## 7. Agent 本机配置

共享的 hook、plugin 和权限配置随仓库提交。模型、endpoint 和 API key 只放本机
配置，不入库：

- Claude Code：`.claude/settings.local.json`
- OpenCode：`.opencode/opencode.json`

API key 不要写入共享 `settings.json`、`config.yaml`、任务文件或日志。
