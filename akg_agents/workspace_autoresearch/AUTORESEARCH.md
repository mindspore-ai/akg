# AutoResearch

AutoResearch 是 Claude Code 驱动的算子优化工作流。它把一个 reference、一个 seed kernel 或 DSL 工程整理成 task_dir，然后按状态机自动完成 baseline、计划、编辑、评测、保留或回滚，直到达到轮数上限并生成 FINISH 报告。

主循环：`scaffold -> BASELINE -> PLAN -> EDIT -> eval -> KEEP/DISCARD -> FINISH`。

本文把 `backend`、`framework`、`dsl` 当作变量说明。示例中的 `<backend>`、`<dsl>`、`<device-id>` 需要按任务实际目标替换；只有在 DSL 专属契约里才点名具体 DSL。

## 快速导航

| 目标 | 看这里 |
|---|---|
| Claude Code 和评测在同一台机器 | [A. 本机运行](#a-本机运行) |
| 本机只跑 Claude，远端机器跑 eval | [B. 远端 worker](#b-远端-worker) |
| 批量跑多个算子 | [C. 批量运行](#c-批量运行) |
| 恢复、重试、看状态 | [D. 恢复与排查](#d-恢复与排查) |
| CLI 参数 | [1. CLI 参数](#1-cli-参数) |
| 文件与命名契约 | [2. 任务文件契约](#2-任务文件契约) |
| DSL 项目格式 | [3. DSL 项目契约](#3-dsl-项目契约) |
| eval 和 worker 链路 | [4. Eval 与 worker 链路](#4-eval-与-worker-链路) |
| 状态机与产物 | [5. 状态机与产物](#5-状态机与产物) |
| env.sh 写法 | [6. env.sh 契约](#6-envsh-契约) |

---

# 操作

## A. 本机运行

适用场景：Claude Code 和 eval 在同一台机器上。机器可以是 Ascend、CUDA 或 CPU 环境，只要目标 backend/DSL 的依赖齐全。

### A.1 准备环境

先确认目标 backend 的基础工具可用：

| backend | 最小自检 |
|---|---|
| `ascend` | `npu-smi info`；`python -c "import torch_npu"` |
| `cuda` | `nvidia-smi`；`python -c "import torch"` |
| `cpu` | `python -c "import torch"` |

如果目标 DSL 有额外运行时依赖，也在同一个环境里检查：Python DSL 检查对应 package import，项目目录类 DSL 检查所需 thirdparty、SDK 和编译工具链。

建议把环境初始化写进 `~/env.sh`，见 [6. env.sh 契约](#6-envsh-契约)。

### A.2 准备 reference 和 kernel

把文件放到 `workspace/` 或任意你能引用到的位置：

```text
workspace/<op>_ref.py
workspace/<op>_kernel.py
```

reference 文件暴露 `Model`、`get_init_inputs()`，并通过 `get_inputs()` 或 `get_input_groups()` 提供单 shape 或多 shape 输入。kernel 文件暴露 `ModelNew`，或者在支持 DSL 工程的任务中指向一个项目目录。

### A.3 启动任务

```bash
cd akg_agents/workspace_autoresearch
source ~/env.sh
claude
```

在 Claude Code 中输入：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

常用可选项：`--max-rounds`、`--eval-timeout`、`--output-dir`、`--worker-url`。目标 backend/framework/DSL 通常来自 `config.yaml: defaults` 或 task config；不要在文档示例里假设某个 DSL 是唯一默认。

### A.4 查看进度

```bash
python scripts/dashboard.py <task_dir> --watch
```

最终产物：

```text
<task_dir>/kernel.py
<task_dir>/.ar_state/report.md
<task_dir>/.ar_state/history.jsonl
```

## B. 远端 worker

适用场景：本机只跑 Claude Code，eval 通过 SSH tunnel 发送到远端评测机器。远端可以是任意支持目标 backend/DSL 的机器。

### B.1 远端准备

远端机器需要有一份 AKG checkout，例如：

```text
/path/to/akg
```

远端 `env.sh` 需要在非交互 shell 中可用。它应完成 Python 环境、硬件 SDK、编译工具链和目标 DSL 依赖的初始化。自检命令按 backend/DSL 选择，不要把某个 backend 的工具当成通用要求。

```bash
source /path/to/env.sh
python -c "import torch"
# backend/DSL-specific checks go here: hardware CLI, Python package imports, SDK/toolchain probes.
```

某些 DSL 需要额外准备 thirdparty 或编译资产；这类步骤应写在 DSL 专属说明或项目 README 里，而不是作为 worker 的通用前置条件。

### B.2 本机配置 SSH 和 worker host

本机 `~/.ssh/config` 中配置 SSH alias，例如：

```text
Host eval-host
    HostName <remote-host-or-ip>
    User <remote-user>
    IdentityFile <optional-private-key>
```

在本机 `config.yaml` 写入：

```yaml
remote_worker:
  hosts:
    eval-host:
      repo_path: /path/to/akg
      env_script: /path/to/env.sh
      ssh_alias: eval-host
```

`ssh_alias` 可省略，默认等于 `remote_worker.hosts` 下的 key。

### B.3 启动 worker 并建立 tunnel

先看状态和远端诊断：

```bash
akg_cli worker --remote-host <alias> --status \
  --backend <backend> --dsl <dsl>
```

启动远端 worker：

```bash
akg_cli worker --remote-host <alias> --start \
  --backend <backend> --devices <device-ids> --dsl <dsl>
```

这条命令会做两件事：

1. SSH 到远端，在 `repo_path/akg_agents` 下运行 `python -m akg_agents.cli.cli worker --start ...`。
2. 在本机建立 `127.0.0.1:<port> -> 远端 127.0.0.1:<port>` 的 SSH tunnel。

默认端口来自 `config.yaml: worker.port`。

验证：

```bash
akg_cli worker --remote-host <alias> --status
```

### B.4 使用远端 worker 跑 /autoresearch

在 Claude Code 中：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --worker-url 127.0.0.1:<port> --devices <device-id>
```

`--devices` 是远端 worker 管理的设备 id。部分批跑链路可以在 `--worker-url` 存在时省略 `--devices`，由 worker 自己分配设备；单任务调试时建议显式传入，方便复现。

停止：

```bash
akg_cli worker --remote-host <alias> --stop
```

## C. 批量运行

batch 目录用于一次跑多个 `(ref, kernel)`。标准布局：

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

如果要走正式 KernelVerifier 链路做 Tier-2 预检：

```bash
python scripts/batch/verify.py <batch_dir> --full --devices <device-id>
```

远端 worker 预检：

```bash
python scripts/batch/verify.py <batch_dir> --full \
  --worker-url 127.0.0.1:<port>
```

批跑：

```bash
python -u scripts/batch/run.py <batch_dir> --devices <device-id> \
  2>&1 | tee -a <batch_dir>/batch.log
```

远端 worker 批跑：

```bash
python -u scripts/batch/run.py <batch_dir> \
  --worker-url 127.0.0.1:<port> --devices <device-id> \
  2>&1 | tee -a <batch_dir>/batch.log
```

监控和汇总：

```bash
python scripts/batch/monitor.py <batch_dir>
python scripts/batch/summarize.py <batch_dir>
```

## D. 恢复与排查

| 场景 | 命令或文件 |
|---|---|
| 恢复最近任务 | `/autoresearch --resume` |
| 恢复指定任务 | `/autoresearch --resume <task_dir>` |
| 看 task 状态 | `<task_dir>/.ar_state/state.json` |
| 看当前 plan | `<task_dir>/.ar_state/plan.md` |
| 看历史轮次 | `<task_dir>/.ar_state/history.jsonl` |
| 看实时进度 | `python scripts/dashboard.py <task_dir> --watch` |
| worker 状态 | `akg_cli worker --remote-host <alias> --status` |
| 重建 tunnel | 再执行一次 `worker --remote-host <alias> --start ...` |

---

# 参考

## 1. CLI 参数

### 1.1 `/autoresearch`

新建任务：

```text
/autoresearch --ref <reference.py> --kernel <kernel.py-or-project-dir> \
  --op-name <op> [--devices <ids>] [--worker-url <host:port>] \
  [--max-rounds N] [--eval-timeout seconds] [--output-dir ar_tasks]
```

恢复任务：

```text
/autoresearch --resume
/autoresearch --resume <task_dir>
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--ref` | reference 文件路径。 |
| `--kernel` | seed kernel 文件，或支持 DSL 的项目目录。 |
| `--op-name` | op 名称，用于 task_dir 前缀和报告。 |
| `--devices` | 本机或远端 worker 的设备 id，逗号分隔。 |
| `--worker-url` | 远端 worker URL，例如 `127.0.0.1:<port>`。 |
| `--max-rounds` | 优化轮数上限，默认来自 `config.yaml: defaults.max_rounds`。 |
| `--eval-timeout` | 单 shape eval 预算，默认来自 `config.yaml: defaults.eval_timeout`。 |
| `--output-dir` | task_dir 父目录，默认 `ar_tasks/`。 |
| `--no-code-checker` | 在该任务的 `task.yaml` 里关闭 CodeChecker。 |

### 1.2 `akg_cli worker`

```bash
akg_cli worker --start  [options]
akg_cli worker --status [options]
akg_cli worker --stop   [options]
```

参数：

| 参数 | 说明 |
|---|---|
| `--backend {ascend,cuda,cpu}` | worker 后端。 |
| `--arch` | 硬件 arch；不传时按 backend 和第一张 device 自动探测。 |
| `--devices` | worker 管理的设备 id，逗号分隔。 |
| `--dsl` | 目标 DSL，用于诊断策略；不同 DSL 的必需运行时依赖会被分开判断。 |
| `--port` | worker 端口，默认 `config.yaml: worker.port`。 |
| `--remote-host` | 使用 `config.yaml: remote_worker.hosts.<alias>` 定义的远端机器。 |

行为要点：

- 本地模式不传 `--remote-host`。
- 远端模式会先做 SSH/env/backend/DSL/端口诊断，再启动远端 daemon 和本机 tunnel。
- `--status` 只探活和诊断，不会启动 daemon。
- `--start` 幂等：worker 已 ready 时直接返回当前状态。

### 1.3 `scripts/batch/run.py`

```bash
python scripts/batch/run.py <batch_dir> [--devices ids] [--worker-url host:port]
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--devices` | 传给每个 task 的设备 id。没有 `--worker-url` 时必填。 |
| `--worker-url` | 整批 eval 走远端 worker。 |
| `--max-rounds` | 覆盖每个 task 的轮数上限。 |
| `--eval-timeout` | 覆盖每个 task 的 eval timeout。 |
| `--timeout-min` | 单个 op 的 wall-clock 上限，默认来自 `batch.run_timeout_min`。 |
| `--only` | 只跑指定 op，逗号分隔。 |
| `--limit` | 最多跑 N 个 op。 |
| `--retry-errored` | 把 error 状态重新纳入队列。 |
| `--cooldown-sec` | 两个 op 之间的等待时间。 |
| `--claude-bin` | Claude CLI 可执行文件。 |
| `--model` | 传给 `claude --model`。 |
| `--extra-claude-arg` | 额外传给 Claude CLI，可重复。 |

## 2. 任务文件契约

### 2.1 Reference

reference 文件是 PyTorch 标准答案，至少包含：

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, *args):
        ...

def get_init_inputs():
    return []

def get_inputs():
    return [arg0, arg1]
```

多 shape 使用 `get_input_groups()`：

```python
def get_input_groups():
    return [
        [arg0_case0, arg1_case0],
        [arg0_case1, arg1_case1],
    ]
```

约定：

- `get_init_inputs()` 用于构造 `Model(*init_inputs)`。
- `get_inputs()` 表示单 shape。
- `get_input_groups()` 表示多 shape；每一组都是一次 forward 的输入。
- batch 模式下文件名必须是 `refs/<op>_ref.py`。

### 2.2 Kernel

常规 Python kernel 文件至少包含：

```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, *args):
        ...
```

约定：

- batch 模式下文件名必须是 `kernels/<op>_kernel.py`。
- scaffold 后会复制到 task_dir 的 `kernel.py`。
- 支持项目目录的 DSL 可以把 `--kernel` 指向工程目录；scaffold 会按 DSL adapter 的规则复制工程。

### 2.3 task_dir 布局

```text
ar_tasks/<op>_<timestamp>_<uuid>/
  reference.py
  kernel.py
  task.yaml
  .ar_state/
    state.json
    plan.md
    history.jsonl
    report.md
```

`task.yaml` 是任务的稳定配置面。下面是字段形状，不是推荐某个 backend/DSL：

```yaml
name: <op>
backend: <backend>
framework: torch
dsl: <dsl>
ref_file: reference.py
kernel_file: kernel.py
devices: [<device-id>]
max_rounds: 30
eval_timeout: 600
worker:
  urls: []
editable_files:
  - kernel.py
```

不要手改 `.ar_state/` 里的运行状态文件，除非是在明确排查损坏状态。

## 3. DSL 项目契约

AutoResearch 通过 `backend/framework/dsl` 三元组选择 adapter。`config.yaml: defaults` 可以给新任务提供默认值，但任务自己的 `task.yaml` 才是运行时最终配置。不要把默认值理解成框架只支持某个 DSL。

字段形状：

```yaml
defaults:
  backend: <backend>
  framework: torch
  dsl: <dsl>
  skill_dsl: <skill-dir-name>
```

常见 DSL：

| DSL | `--kernel` 形态 | 可编辑范围 |
|---|---|---|
| `triton_ascend` / `triton_cuda` | 单个 Python 文件 | 通常是 `kernel.py`。 |
| `ascendc` | direct-invoke AscendC 工程目录或包装后的 kernel 文件 | `task.yaml: editable_files` 列出的算子源文件；不要改构建脚本和无关资产。 |
| `ascendc_catlass` | CATLASS 风格 C++/AscendC 工程 | 算子实现文件和必要的 wrapper；CATLASS thirdparty 不应由 agent 修改。 |
| `cuda_c` / `cpp` | C/C++ kernel 包装文件或工程 | adapter 列出的源码文件。 |
| `pypto` | Python 文件 | `kernel.py`。 |
| `tilelang_cuda` / `tilelang_ascend` | Python/TileLang 文件 | `kernel.py` 或 adapter 声明的文件。 |

项目目录类 DSL 的原则：

- adapter 负责识别构建入口、复制工程、声明 `editable_files`。
- agent 只能编辑 `editable_files` 内的文件。
- 工程里的样例脚本、下载脚本、third_party、build 目录默认不是优化目标。
- 如果需要扩大可编辑范围，应通过 task config 或 DSL adapter 明确表达，不要靠提示词临时放开。

## 4. Eval 与 worker 链路

当前正式链路：

```text
baseline.py / pipeline.py
  -> task_config.run_eval(...)
  -> utils.akg_eval.eval_kernel(...)
  -> eval.KernelVerifier
```

本地 eval 和远端 worker eval 共用同一条 `KernelVerifier` 路径。worker 只是把包发到远端机器执行，不再维护另一套评测语义。

### 4.1 本地 eval

```text
run_eval(task_dir, config, device_id=<device-id>)
  -> KernelVerifier local backend
  -> verify_result.json / profiling metrics
```

### 4.2 远端 eval

```text
run_eval(task_dir, config, worker_urls=["127.0.0.1:<port>"])
  -> RemoteWorker HTTP client
  -> remote worker.server
  -> KernelVerifier on remote host
```

worker daemon 暴露的主要端点：

| 端点 | 用途 |
|---|---|
| `/api/v1/status` | daemon 状态、backend、arch、devices、log_file。 |
| `/api/v1/health` | 非阻塞探测 device pool/event loop 是否健康。 |
| `/api/v1/verify` | 执行 verify。 |
| `/api/v1/profile` | 执行 profile。 |
| `/api/v1/acquire_device` / `/api/v1/release_device` | worker 侧设备分配。 |

### 4.3 baseline 和指标

- PyTorch reference latency 是 baseline source。
- seed/candidate kernel 的 correctness 和性能都通过 `KernelVerifier` 产物进入决策。
- 多 shape 任务会记录 per-shape 描述和 shape signature，用于 sticky baseline 和漂移检测。
- `batch/verify.py --full` 也走正式 `KernelVerifier` verify-only 路径。

### 4.4 CodeChecker

`quick_check.py` 在每轮 eval 前先扫 `task.yaml: editable_files` 中的 Python 文件，目的是提前拦住“看起来能跑、实际绕过 DSL 或污染验证”的改动，避免浪费一次完整评测。

它主要检查四类问题：

| 类别 | 会拦什么 |
|---|---|
| Python 基础质量 | `ast.parse` / `py_compile` 失败、关键 import 不可解析、代码 token 中混入连续 CJK 文本。 |
| Kernel launch 退化 | 声明了 JIT/kernel DSL，但没有对应 kernel，或 kernel 没有被 launch，或 `forward()` 仍用 torch 高层核心算子完成主要计算。 |
| DSL 入口退化 | 需要显式入口的 DSL 必须真实调用自己的入口，不能只保留包装层而把主计算交给 framework API。 |
| autotune 安全 | 使用 autotune 时必须恢复输出 buffer，避免不同 config 之间互相污染。 |

硬黑名单是“核心计算不能留给 torch”的算子，例如 matmul/conv/einsum/softmax/embedding/FFT/solve 等；软黑名单是 relu/exp/sum/mean/pool/norm/interpolate 这类可能作为融合前后处理出现的算子：如果已经真正 launch 了 DSL kernel，软项只记录为提示，否则会被视为退化。

任务级关闭只应用在明确知道规则误伤时，例如临时迁移 seed 或非 Python DSL 包装不适合 AST 扫描：

```yaml
code_checker:
  enabled: false
```

关闭后 `quick_check.py` 只保留 editable file 存在性检查和可选 smoke test，不再做 DSL 反作弊检查；正式正确性和性能仍由 `KernelVerifier` 决定。

规则入口在 `scripts/utils/config/code_checker.yaml`；新增 DSL 规则时应扩展 `scripts/utils/code_checker.py` 的独立 compliance check，而不是把 DSL 特判塞进通用流程。

## 5. 状态机与产物

### 5.1 Phase

```text
BASELINE
  -> PLAN
  -> EDIT
      -> quick_check
      -> eval
      -> KEEP or DISCARD or FAIL
  -> REPLAN / DIAGNOSE / FINISH
```

关键规则：

- `BASELINE` 初始化 baseline 和 seed 状态。
- `PLAN` 由 `create_plan.py` 写入并校验 `plan.md`。
- `EDIT` 中 agent 修改 `editable_files`，然后 `pipeline.py` 执行 quick_check 和 eval。
- `KEEP` 会保留改动、更新 best，并写入历史。
- `DISCARD` 会回滚本轮 editable files。
- 连续失败达到阈值后进入 `DIAGNOSE`。
- 达到 `max_rounds` 后进入 `FINISH`，生成报告。

### 5.2 产物

| 文件 | 说明 |
|---|---|
| `.ar_state/state.json` | phase、owner、progress、best、失败计数等。 |
| `.ar_state/plan.md` | 当前计划和 plan item 状态。 |
| `.ar_state/history.jsonl` | 每轮 eval/decision/metrics/commit 记录。 |
| `.ar_state/report.md` | FINISH 报告。 |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 子代理产物。 |
| `.ar_state/plan_items.xml` | 给 `create_plan.py` 的结构化计划输入。 |

### 5.3 batch 产物

```text
<batch_dir>/
  manifest.yaml
  batch_progress.json
  batch.log
  verify_results.json
  refs/<op>_ref.py
  kernels/<op>_kernel.py
```

`batch_progress.json` 记录每个 op 的状态、task_dir、指标和错误信息。

## 6. env.sh 契约

`env.sh` 的唯一职责：source 完成后，当前 shell 的 `python` 能 import 目标 backend/DSL 所需包，且硬件工具在 PATH 中。

模板：

```bash
# Optional: initialize package manager / virtual env.
source /path/to/conda.sh
conda activate <env-name>

# Optional: initialize hardware SDK, compiler, and runtime paths.
source /path/to/sdk/set_env.sh
```

按任务目标选择验证项：

| 目标 | 验证 |
|---|---|
| Python/framework | `python -c "import torch"` |
| Ascend backend | `python -c "import torch_npu"`；`npu-smi info` |
| CUDA backend | `nvidia-smi` |
| Triton-family DSL | `python -c "import triton"` |
| 项目目录类 DSL | 运行该项目 README 或 adapter 要求的构建前置检查。 |

远端 worker 会在非交互 SSH shell 中执行：

```bash
source <env_script>
cd <repo_path>/akg_agents
akg_cli worker --start ...
```

所以不要依赖交互 shell 专属的 alias 或手动 cd。

## 7. Skills

`skills/` 下按 DSL 组织优化知识。`config.yaml: defaults.skill_dsl` 决定 hook 在 PLAN 阶段提示 Claude 查找哪一类 skill。

常见目录示例，实际以仓库内容为准：

```text
skills/<dsl-or-domain>/
skills/performance-summary/
skills/task-constructor/
```

PLAN 阶段应该读取 1-3 个最相关的 `SKILL.md`，并把使用到的文件名写进 plan rationale。

## 8. 并发与恢复建议

| 资源 | 建议 |
|---|---|
| 一个 task_dir | 同时只让一个 Claude session 驱动。 |
| 一个 batch_dir | 同时只跑一个 `scripts/batch/run.py`。 |
| 一张设备卡 | 交给一个本地 eval 或一个 worker 管理；多任务通过 worker 队列复用。 |
| 修改 worker 代码后 | 重启 worker，确保远端 daemon 读到新代码。 |
| tunnel 断开 | 重新执行 `worker --remote-host <alias> --start ...`。 |

常见故障：

| 现象 | 处理 |
|---|---|
| `worker --status` unreachable | 看诊断输出；检查 SSH、env_script、端口占用和远端日志。 |
| `batch` 卡在 running | 确认子进程已退出后用 `--retry-errored` 重跑。 |
| CodeChecker 拦截 seed | 临时用 `--no-code-checker` scaffold，或在 task.yaml 中关闭。 |
| 远端返回旧代码行为 | 远端 repo pull 到最新并重启 worker。 |
| 正确性失败 | 查看 `verify_result.json`、失败日志和 failure extractor 摘要。 |
