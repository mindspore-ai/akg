# workspace_autoresearch

Claude Code 驱动的算子自动优化框架。本目录是 `akg_agents/` 的使用态工作空间：
phase machine + hooks + slash command 在 `scripts/`，verifier 与 worker 走
`utils.akg_eval` 桥（直接 import `akg_agents.op.verifier.KernelVerifier` +
`akg_agents.core.worker.manager`，不 vendor）。

**主循环**：scaffold → BASELINE → PLAN → EDIT → eval → KEEP/DISCARD → 直到 `max_rounds` → FINISH。

**后端**：默认 Ascend NPU（[`config.yaml`](config.yaml) `defaults.backend: ascend`、`dsl: triton_ascend`、`skill_dsl: triton-ascend`）。下文 walkthrough 以 NPU 为典型，但 workspace 是 single-target-per-repo 设计，**改 `config.yaml` 的 `defaults` triple 即可切到 cuda / cpu**：scaffold 会按 `defaults.backend` 走 `nvidia-smi` / `npu-smi` / `platform.machine` 自动推断 `arch`，verifier / worker dispatch 也走对应路径。具体可切换的 DSL 见 [skills 根目录](../python/akg_agents/op/resources/skills/) 的子目录名（triton-ascend / triton-cuda / pypto / cpp / cuda-c / tilelang-cuda 等）。

第一次用：按下表选一条「操作」路径走完即可，配置说明都在「参考」里按主题归档。

| 目标 | 跳转 |
|---|---|
| 第一次用，本机就是 NPU 机 | [操作 A](#a-单机本机--测评机) |
| 第一次用，本机没 NPU，要用远端 NPU | [操作 B](#b-双机本机跑-claude无-npu-远端-npu-机跑-eval) |
| 一次跑很多算子 | [操作 C](#c-批量) |
| 续跑、重置、查历史 | [操作 D](#d-续跑--重置--查询) |
| 查 CLI 参数 | [参考 §1](#1-cli-参数) |
| 算子/文件该怎么命名 | [参考 §2](#2-命名契约) |
| 单轮 phase 怎么走 | [参考 §3](#3-主循环--状态机) |
| eval 怎么跑、worker 怎么联 | [参考 §4](#4-eval-执行链) |
| 精度判定标准 | [参考 §5](#5-精度容差) |
| 状态文件在哪 | [参考 §6](#6-文件与状态布局) |
| env.sh 该写什么 | [参考 §7](#7-envsh-契约) |
| 起停 worker daemon / SSH tunnel | [参考 §8](#8-akg_cli-worker) |
| Triton 调优 markdown 库 | [参考 §9](#9-skills-库) |
| 内部机制入口 | [参考 §10](#10-hook-与内部机制) |
| 并发跑 / 多 session / 共用卡 | [参考 §11](#11-并发与冲突) |

---

# 操作

## A. 单机：本机 = 测评机

开发与 eval 同一台 Ascend NPU 机，配置最少。

### A.1 准备 env.sh

前置：本机有 Ascend NPU，`npu-smi info` 列得出设备；本机 Python 已能 `import torch_npu, triton`；Claude Code CLI 已装。

写 `~/env.sh`，使非交互 shell `source` 后即可 `python -c "import torch_npu, triton"`。模板与禁项见 [参考 §7](#7-envsh-契约)。

验证：
```bash
source ~/env.sh && python -c "import torch_npu, triton" && npu-smi info
```

### A.2 写 ref 和 kernel

在 `workspace/` (相对 workspace_autoresearch 根) 下放两个文件。命名硬约束见 [参考 §2](#2-命名契约)。

`workspace/<op>_ref.py`：PyTorch 标准答案，暴露 `class Model(nn.Module)`、`get_inputs()`（或 `get_input_groups()`）和 `get_init_inputs()`。

`workspace/<op>_kernel.py`：种子 kernel，暴露 `class ModelNew(nn.Module)`，forward 内实际调用目标 DSL kernel。命名与接口细节见 [参考 §2](#2-命名契约)。

### A.3 启动

```bash
cd workspace_autoresearch
source ~/env.sh
claude
```

在 Claude 输入框里：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <NPU-id>
```

`--devices` 必填。其他可调 flag（`--max-rounds`、`--eval-timeout` 等）见 [参考 §1.1](#11-autoresearch)。

### A.4 看进度

Claude 全自动按 `scaffold → BASELINE → PLAN → EDIT → ...` 推进，无须人工干预。另开终端实时查看：

```bash
cd workspace_autoresearch
python scripts/dashboard.py <task_dir> --watch
```

阶段流转与产物见 [参考 §3](#3-主循环--状态机)。

### A.5 拿结果

达到 `--max-rounds` 自动进入 FINISH，落盘：
- `<task_dir>/kernel.py`：最佳 kernel
- `<task_dir>/.ar_state/report.md`：报告（含内嵌 SVG）

每次 KEEP 对应一次 git commit。

---

## B. 双机：本机跑 claude（无 NPU）+ 远端 NPU 机跑 eval

本机没有 NPU，eval 委托远端 Ascend 机，orchestrator 留本机。worker 是 HTTP daemon，绑 `127.0.0.1`，访问通过 `akg_cli worker --remote-host` 自动建立的 SSH tunnel，不暴露公网。

### B.1 [两端] 准备环境

| 在哪 | 做什么 |
|---|---|
| 远端 NPU 机 | 准备 AKG checkout 作为 `<repo_path>`；写 `env_script`，source 后 `python -c "import torch_npu, triton"` 成功；`npu-smi info` 可列设备。若使用 `ascendc_catlass`，在 `<repo_path>/akg_agents` 下完成 `bash download.sh --with_catlass`。 |
| 本机 | (1) 装 Python ≥ 3.10、PyYAML、Claude Code CLI。(2) 作为 orchestrator / SSH tunnel / 任务文件所在机器，eval 全部走远端 worker。(3) `~/.ssh/config` 配好 alias（下文以 `my-npu` 为例）+ 密钥免密登录远端。 |

<details><summary><code>~/.ssh/config</code> 怎么配 + 跑前自检</summary>

**第一步：本机 `~/.ssh/config` 配远端别名**。下面用 `my-npu`。

```text
# 文件路径：本机 ~/.ssh/config（没有就新建，注意是 config 不是 config.txt）
Host my-npu                            # 本地 SSH 别名，可按团队习惯命名
    HostName 192.168.x.x               # 远端机真实 IP 或域名
    User <remote-user>                 # 登录远端用的账号，比如 root
    # 私钥放在本机非默认位置时填写：
    # IdentityFile <本机私钥绝对路径>
    # 如需跳板机：
    # ProxyJump bastion
```

`Host` 别名通常与 B.2 的 `remote_worker.hosts.<alias>` 保持一致；需要不同名称时，在 B.2 填 `ssh_alias`。

**第二步：跑前自检**。在本机执行：

```bash
# 命令在本机敲。ssh 后面单引号里那一串都是在远端执行的，所以路径里的 <远端账号> 指远端机
# 的用户名，env.sh 是远端机上的脚本。
ssh my-npu 'source /home/<远端账号>/env.sh \
    && python -c "import torch_npu, triton" \
    && npu-smi info | head -3'
```

这条命令同时验证 SSH、`env_script`、`torch_npu/triton` import 和 NPU 可见性。失败时优先修对应的 SSH alias、env.sh 或 CANN/Python 环境。

</details>

### B.2 [本机] 配 worker host

`config.yaml`：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path:  /home/<user>/akg               # 远端 akg checkout 路径（必填）
      env_script: /home/<user>/env.sh            # 必填，source 后 PATH 上的 python 可 import torch_npu/triton
      ssh_alias:  my-npu                         # 可选，默认 = key
```

`repo_path` 指远端 akg checkout。远端命令会在 `source env_script && cd repo_path/akg_agents` 之后运行 `python -m akg_agents.cli.cli worker ...`；解释器由 `env_script` 配好的 PATH 决定。字段语义见 [参考 §8](#8-akg_cli-worker)。`akg_cli` 读 cwd 的 `./config.yaml`。

### B.3 [本机] 启动远端 worker（akg_cli 自动开 SSH tunnel）

```bash
akg_cli worker --remote-host my-npu --start \
    --backend ascend --arch ascend910b3 --devices <NPU-id>
```

`--port` 可省 —— akg_cli 从 cwd 的 `./config.yaml` 读 `worker.port`；yaml 没设才回落硬编码 9001。一条命令做两件事：(1) SSH 到远端执行 checkout 内的 `python -m akg_agents.cli.cli worker --start ...`，PYTHONPATH 锁到 `repo_path/akg_agents/python`，daemon 跟随该 checkout 启动（见 [§8](#8-akg_cli-worker)），(2) 本机起 `ssh -L <port>:127.0.0.1:<port>`，tunnel pid 存 `~/.akg_agents/tunnels/<port>.pid`。后续 `--worker-url 127.0.0.1:<port>` 透传。

`--start` 幂等：daemon / tunnel 已活时直接返回当前状态；仅 tunnel 退出且 daemon 仍可用时只重建 tunnel；两者都不可用时执行 SSH spawn。tunnel 异常退出后再次执行 `--start` 即可恢复。

多租户机器：默认 9111 经常被其他进程占用，把 `config.yaml: worker.port` 改成 9112+，并在启动命令与 `--worker-url` 里统一使用该端口。

验证：

```bash
akg_cli worker --remote-host my-npu --status
# → {"status":"ready","backend":"ascend","arch":"ascend910b3","devices":[0]}
```

### B.4 [本机] 写 ref 和 kernel

同 [A.2](#a2-写-ref-和-kernel)。

### B.5 [本机] 启动 /autoresearch

```bash
cd workspace_autoresearch && claude
```

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <远端 NPU-id> --worker-url 127.0.0.1:9111
```

`--devices` 取**远端**卡下标。`--worker-url` 由 scaffold 透传写入 `task.yaml: worker.urls`，后续每轮 eval 自动走远端。

### B.6 [本机] 看进度 / 拿结果

同 [A.4](#a4-看进度)、[A.5](#a5-拿结果)。

### B.7 [本机] 停 worker（不再用时）

```bash
akg_cli worker --remote-host my-npu --stop --port 9111
```

先 SIGTERM 本机 tunnel（按 `~/.akg_agents/tunnels/9111.pid`），再 SSH `lsof -ti :9111 | xargs -r kill` 终止远端 daemon。`lsof -ti` 只取占该端口的 PID，避免影响其他进程。

---

## C. 批量

`scripts/batch/` 对一批 `(ref.py, kernel.py)` 任务循环跑 `/autoresearch`。批级状态写 `<batch_dir>/batch_progress.json`，round 级仍在 `ar_tasks/`。

### C.1 标准流程

约束：cwd 在 `autoresearch/` 内；长跑用 tmux/screen 包；`--devices` 指定本机或远端卡。

```bash
BATCH_DIR=/tmp/batch_001
DEVICE=0

# 0. 进入子目录
cd workspace_autoresearch

# 1. 放 ref/kernel（命名必须严格 <op>_ref.py / <op>_kernel.py，见参考 §2）
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp workspace/*_ref.py    $BATCH_DIR/refs/
cp workspace/*_kernel.py $BATCH_DIR/kernels/

# 2. Tier-1 预检：纯静态（语法 / import / 必备 export），任意机器可跑，覆盖 NPU/torch_npu 环境之外的检查
python scripts/batch/prepare.py $BATCH_DIR

# 3.（可选）Tier-2 预检：在有 NPU 的机器上跑 ref vs kernel
python scripts/batch/verify.py $BATCH_DIR --full

# 4. 后台跑（tmux 默认非 login shell，须 bash --login + source env.sh）
tmux new -d -s ar_batch \
    "bash --login -c 'source ~/env.sh && \
     python -u scripts/batch/run.py $BATCH_DIR --devices $DEVICE 2>&1 | tee -a $BATCH_DIR/batch.log'"

# 5. 另开终端监控
python scripts/batch/monitor.py $BATCH_DIR

# 6. 结束后汇总
python scripts/batch/summarize.py $BATCH_DIR
```

**多文件 DSL（ascendc_catlass）的 step 1 差异**：`kernels/` 改成 per-op 子目录布局，整个项目目录跟 sibling python wrapper 一起放：

```bash
# step 1 (catlass 版)：每个 op 一个子目录，含 kernel.py + catlass_op/
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels/<op>
cp workspace/<op>_ref.py    $BATCH_DIR/refs/
cp workspace/<op>/kernel.py $BATCH_DIR/kernels/<op>/        # 或 <op>_kernel.py
cp -r workspace/<op>/catlass_op $BATCH_DIR/kernels/<op>/    # 整个 pybind 项目
```

`prepare.py` / `verify.py` / `run.py` 全部沿用相同命令；manifest 根据 `config.yaml: defaults.dsl` 对应 adapter 的 `kernel_arg_is_directory` 标志自动切换解析路径（见 §6 末尾的目录布局图）。

### C.2 配合远端 worker 跑批（本机无 NPU）

如果本机不是 NPU 机（即 B 章场景），批跑改动很小，整体仍按 [C.1](#c1-标准流程) 操作，但有两处差异：

1. **先按 [B.3](#b3-本机-启动远端-workerakg_cli-自动开-ssh-tunnel) 启动远端 worker 并建立本机 tunnel**（每次开新 batch 前确认 tunnel 还活着：`akg_cli worker --remote-host my-npu --status --port 9111`）。
2. **`run.py` 加 `--worker-url 127.0.0.1:9111`**，本机 tmux 直接跑 `run.py`；环境初始化放在远端 `env_script`，eval 全走远端 worker。

第 4 步的 tmux 命令变为：

```bash
tmux new -d -s ar_batch \
    "python -u scripts/batch/run.py $BATCH_DIR \
       --worker-url 127.0.0.1:9111 --devices $REMOTE_DEVICE \
       2>&1 | tee -a $BATCH_DIR/batch.log"
```

`--devices` 在有 `--worker-url` 时仍要给，但取值是**远端**卡下标（透传到每个 op 的 task.yaml）。

其它步骤（prepare / monitor / summarize）和 [C.1](#c1-标准流程) 一样在本机直接跑。Tier-2 verify 需要 NPU 环境，完整预检可在远端执行 `python scripts/batch/verify.py <batch_dir> --full`；常规批跑也会在每轮 eval 中完成 verify。

### C.3 监控工具一览

| 工具 | 数据源 | 用途 |
|---|---|---|
| `monitor.py` | progress + ar_tasks/ | 实时队列 + phase + heartbeat |
| `monitor.py --dashboard` | 同上 | `execvp` 至 `dashboard.py` 看当前 task TUI |
| `summarize.py` | 仅 progress JSON | 离线汇总，复制粘贴友好 |
| `tail -f batch.log` | claude stdout | 查 hook 输出、Edit、Bash |

### C.4 断点续跑 / 重试

- `done` 保持完成状态
- `error` 默认跳过；`--retry-errored` 重新纳入
- `pending` 自动续
- `running`（终止瞬间在跑的 op）下次启动时自动降级为 `error`
- transient API/network 失败且 task 状态完整时，batch supervisor 会按 `config.yaml: batch.transient_retries` 接续

同一 batch_dir 由 `.batch.lock` 串行化；已退出进程残留的 lock 会在下次启动时自动判活清理。

完整 batch CLI 参数见 [参考 §1](#1-cli-参数)。

---

## D. 续跑 / 重置 / 查询

A/B/C 均适用：

| 操作 | 命令 |
|---|---|
| 续跑最近 task | `/autoresearch --resume` |
| 续跑指定 task | `/autoresearch --resume <task_dir>` |
| 重新开始 | 删除 `ar_tasks/<task_dir>/`，再 `/autoresearch --ref ... --kernel ...` |
| 查每轮记录 | `<task_dir>/.ar_state/history.jsonl`（一行一轮）|
| 查 plan 当前状态 | `<task_dir>/.ar_state/plan.md` |

---

---

# 参考

操作部分已经涵盖正常使用。下面按主题归档配置和机制说明。

## 1. CLI 参数

### 1.1 `/autoresearch`

四种调用形态由参数决定：

| 形态 | 触发条件 | 行为 |
|---|---|---|
| 新建任务 | `--ref X.py --kernel Y.py --op-name N` + (`--devices` 或 `--worker-url`) | scaffold 建 task_dir → 跑 baseline → 进 PLAN |
| 续跑最近 | `--resume`（无目录参数）| 自动找最近活跃的 task |
| 续跑指定 | `--resume <task_dir>` 或直接 `<task_dir>` | 续指定目录 |

新建任务的完整参数：

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--ref` | 路径 | ✅ | — | PyTorch ref 文件（`<op>_ref.py` 或任意 `.py`，详见 [§2](#2-命名契约)）|
| `--kernel` | 路径 | ✅ | — | 种子 kernel 文件 |
| `--op-name` | 字符串 | ✅ | — | op 名，决定 task_dir 前缀 |
| `--devices` | 整数或逗号列表（`5` 或 `0,1,2,3`）| 二选一 | — | 本机 NPU 卡 id；如果给了 `--worker-url` 则可不填，由 worker 自带卡列表 |
| `--worker-url` | `host:port[,host:port]` | 二选一 | — | 走远端 worker（写入 `task.yaml: worker.urls`）|
| `--max-rounds` | int | ❌ | 30（[config.yaml](config.yaml): `defaults.max_rounds`） | 优化轮数上限，触发后进 FINISH |
| `--eval-timeout` | int 秒 | ❌ | 600（同上 `defaults.eval_timeout`） | 单 shape 的 verify+profile wall-clock 上限 |
| `--output-dir` | 路径 | ❌ | `ar_tasks` | task_dir 父目录 |
| `--no-code-checker` | flag | ❌ | (启用) | 关掉 [`engine/quick_check.py`](scripts/engine/quick_check.py) 的 Triton 退化 AST 检查（一般用于调试种子 kernel）|

注：`arch`（如 `ascend910b3` / `a100` / `rtx4060`）由 `--devices` 选中的卡经 backend-appropriate 探测工具自动推断（ascend → `npu-smi info`，cuda → `nvidia-smi --query-gpu=name`，由 [`config.yaml`](config.yaml) 的 `defaults.backend` 决定走哪条），写进 `task.yaml` 仅供 dashboard / report 显示；用户传入时覆盖自动推断值。

---

### 1.2 `akg_cli worker` （worker 启停 + tunnel）

AKG canonical CLI。三种模式互斥：

| 模式 | 用法 |
|---|---|
| `--start` | 启动 worker daemon。如配合 `--remote-host`，会先 SSH 到远端执行 checkout 内的 `python -m akg_agents.cli.cli worker --start ...`（PYTHONPATH 锁到 `repo_path/akg_agents/python`），再在本机起 `ssh -L <port>:127.0.0.1:<port>` tunnel |
| `--stop` | 停 `--port` 上的 daemon；配合 `--remote-host` 时先 SIGTERM 本机 tunnel，再 SSH `lsof -ti :<port> \| xargs kill` 杀远端 |
| `--status` | `curl 127.0.0.1:<port>/api/v1/status`；配合 `--remote-host` 走的是本机 tunnel（假定 `--start` 已建好）|

主要 flag：

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--start` / `--stop` / `--status` | flag | ✅ 三选一 | — | 互斥。`--start` 幂等；`--status` 纯查询 |
| `--backend` | `ascend` / `cuda` | `--start` 时必填 | — | 硬件后端 |
| `--arch` | 字符串（如 `ascend910b3`） | ❌ | `a100` | 写入 worker 自报的 arch；远端启动可显式传以覆盖 |
| `--devices` | 逗号分隔 | ❌ | `0` | worker 管理的卡集合 |
| `--port` | int | ❌ | `./config.yaml: worker.port`，否则 9001 | TCP 端口；WA 习惯用 9111+ |
| `--remote-host` | SSH alias | ❌ | (本机) | 通过 SSH 在 `./config.yaml: remote_worker.hosts.<alias>` 定义的远端机执行；`--start` 额外打通本机 tunnel |

tunnel pid 落在 `~/.akg_agents/tunnels/<port>.pid`；远端 daemon log 落在远端 `/tmp/akg_worker_<port>.log`。

---

### 1.3 `scripts/batch/run.py`

位置参数 `<batch_dir>` 必填（指向 [§6](#6-文件与状态布局) 的 batch 目录）。

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--devices` | 整数或逗号列表 | ✅（除非给了 `--worker-url`）| — | 透传给每个 op 的 `/autoresearch --devices` |
| `--worker-url` | `host:port[,host:port]` | ❌ | — | 透传 `--worker-url`，整批走远端 worker，本机可无 NPU |
| `--mode` | `ref-kernel` / `ref` | ❌ | `manifest.mode` 或 `ref-kernel` | 整批是否要 kernel；`ref` 模式下 kernel 由 agent 现写 |
| `--max-rounds` | int | ❌ | 同 `/autoresearch` | 透传 per-op |
| `--eval-timeout` | int 秒 | ❌ | 同上 | 透传 per-op |
| `--timeout-min` | int 分钟 | ❌ | 180（[config.yaml](config.yaml): `batch.run_timeout_min`）| 单 op wall-clock 上限，超时杀 `claude --print` 子进程 |
| `--only` | 逗号分隔 op 名 | ❌ | — | 队列过滤：只跑这些 |
| `--limit` | int | ❌ | — | 队列过滤：跑前 N 个 |
| `--retry-errored` | flag | ❌ | (跳过) | 把 `error` / `stale running` 重新纳入队列（下一次批跑时生效）|
| `--cooldown-sec` | int 秒 | ❌ | 5（[config.yaml](config.yaml): `batch.cooldown_sec`）| 两个 op 之间的间隔 |
| _无 CLI flag_ | int | — | 3（[config.yaml](config.yaml): `batch.transient_retries`）| 单 op 内 transient 自动续：claude.exe 异常退但 framework progress 完整时，supervisor 自动 `/autoresearch --resume --force` 接续最多 N 次。见 [§C.4](#c4-断点续跑--重试)。|
| `--claude-bin` | 路径 | ❌ | `claude` | Claude CLI 可执行文件路径（Windows 下需指向 `claude.cmd`）|
| `--model` | 字符串 | ❌ | (空) | 透传到 `claude --model`，覆盖默认模型 |
| `--extra-claude-arg` | 字符串（可重复）| ❌ | — | 任意额外 flag 透传给 `claude --print`（可叠加：`--extra-claude-arg --foo --extra-claude-arg --bar`）|

## 2. 命名契约

### 2.1 跨 DSL 通用

| 项 | 约束 |
|---|---|
| ref 文件名 | 单跑任意；批跑严格 `<op>_ref.py` |
| kernel 文件名 | 单跑任意；批跑严格 `<op>_kernel.py`（ascendc_catlass 例外，见 §2.2） |
| ref 暴露 | `class Model(nn.Module)` + `get_init_inputs()`，并二选一：`get_inputs()` 单 shape / `get_input_groups()` 多 shape |
| 同目录数据文件 | 与 ref 同名或以 ref basename + `_` 开头的 `.json` / `.pt` / `.npz` 会随任务打包。例如 `31_IOU.py` 可配 `31_IOU.json` 或 `31_IOU_cases.json` |
| task_dir 命名 | scaffold 生成 `ar_tasks/<op_name>_<unix_ts>_<uuid6>/`，供 resume、dashboard、batch 关联 |
| 重命名约定 | scaffold 在 task_dir 内落地为 `reference.py` / `kernel.py`；KEEP commit 按 `task.yaml.editable_files` stage |

### 2.2 各 DSL 的 kernel 契约

`defaults.dsl`（[config.yaml](config.yaml)）选定后，整库走对应 DSL 的契约。使用时主要关注 kernel 入口和 `--kernel` 形态：

| DSL | kernel 入口 | `--kernel` 形态 | 编译 |
|---|---|---|---|
| `triton_ascend` / `triton_cuda` | `class ModelNew(nn.Module)`，forward 调用 `@triton.jit` kernel | 单 `.py` | 否 |
| `pypto` | `<op>_pypto_<framework>(...)` + `@pypto.jit` kernel | 单 `.py` | 否 |
| `tilelang_cuda` / `tilelang_npuir` | `<op>_tilelang_*_<framework>(...)` 函数 | 单 `.py` | 否 |
| `cpp` / `cuda_c` | `class ModelNew(nn.Module)`，源码字符串交给 inline toolchain | 单 `.py` | 运行时 |
| `ascendc` | `<op>_kernel(...)` 函数 | 单 `.py` | 是 |
| `ascendc_catlass` | `class ModelNew(nn.Module)` 调 `torch.ops.<ns>.<entry>(...)` | `catlass_op/` 目录 | 是 |
| `swft` | `<op>_swft_<framework>(...)` 函数 | 单 `.py` | 否 |

详细 adapter 和检查规则在 [`python/akg_agents/op/verifier/adapters/dsl/`](../python/akg_agents/op/verifier/adapters/dsl/) 与 [`code_checker.py`](../python/akg_agents/op/utils/code_checker.py)。

<details><summary>ascendc_catlass 目录模板</summary>

**目录布局**（推荐 per-op 子目录，方便多 op 共存）：

```
workspace/
├── <op>_ref.py                            ← PyTorch 标准答案（同 Triton）
└── <op>/                                  ← per-op 子目录
    ├── kernel.py  (或 <op>_kernel.py)     ← (1) Python ModelNew wrapper
    └── catlass_op/                        ← pybind project
        ├── kernel/catlass_kernel.asc      ← (2) AscendC 模板核
        ├── include/catlass_kernel.h       ← (3) C 头：kernel entry 声明
        ├── src/catlass_torch.cpp          ← (4) pybind：注册 torch.ops.<ns>.*
        └── CMakeLists.txt                 ← (5) cmake（用 CATLASS_ROOT）
```

`/autoresearch --kernel` 指向 **`catlass_op/` 目录**；scaffold 在其父目录里找 `kernel.py`，找不到再退到 `<op_name>_kernel.py`（[`ascendc_catlass.py::read_kernel_source`](../python/akg_agents/op/verifier/adapters/dsl/ascendc_catlass.py)），两者都不在直接报错退出。

scaffold 自动写入这些 editable files：

```yaml
editable_files:
  - kernel.py
  - catlass_op/kernel/catlass_kernel.asc
  - catlass_op/include/catlass_kernel.h
  - catlass_op/src/catlass_torch.cpp
  - catlass_op/CMakeLists.txt
```

CATLASS 头文件默认来自 `<akg-root>/thirdparty/catlass`，也可由 `CATLASS_ROOT` 或 `task.yaml: catlass.root` 覆盖。

</details>

## 3. 主循环 / 状态机

单轮：`PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle`。
连续 N 次 FAIL 转 `DIAGNOSE`（N 默认 3，配 [`config.yaml`](config.yaml) `defaults.consecutive_fail_threshold`）；plan 全部 settle 转 `REPLAN`；预算耗尽转 `FINISH`。DIAGNOSE 子代理对同一 plan_version 最多重试 `defaults.diagnose_max_attempts` 次（默认 5），用尽后转人工 fallback。

```
INIT
  │  /autoresearch --ref X.py --kernel Y.py
  ▼
BASELINE  (scaffold --run-baseline 原子完成，运行 seed kernel)
  │
  ▼
PLAN  (BASELINE PASS 或 FAIL 均进入 PLAN；FAIL 时首批 plan items 改写 seed)
  │  create_plan.py 校验
  ▼
   ┌────────────────────────────────── EDIT ◀────────────┐
   │  pipeline.py:                                       │
   │    quick_check → run_eval → keep_or_discard         │
   │   ├─ KEEP    : git commit (editable_files)，更新 best│
   │   ├─ DISCARD : 回滚 editable_files                   │
   │   └─ FAIL    : consecutive_failures++，回滚         │
   │                                                     │
   │   ├─ failures ≥ N (config)     ─→ DIAGNOSE ─→ PLAN ─┤
   │   ├─ plan 全部 settle           ─→ REPLAN  ─→ PLAN ─┤
   │   └─ eval_rounds == max_rounds ─→ FINISH
   └─────────────────────────────────────────────────────┘
```

DIAGNOSE / REPLAN 不绕回 PLAN：`create_plan.py` 校验通过后 hook 直接写 `phase = EDIT`。每个 plan item 在 `history.jsonl` 中持有 KEEP/DISCARD/FAIL 终态，或在 REPLAN/DIAGNOSE 边界被丢弃；pid 单调递增不复用。

阶段产物：

| 阶段 | Claude 操作 | 产物 |
|---|---|---|
| BASELINE | `baseline.py` | seed_metric 写入 state.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 (ACTIVE) 标记）|
| EDIT | Edit kernel.py 后跑 `pipeline.py` | history.jsonl，可选 git commit |
| FINISH | (auto) `pipeline.py` → `report.py` | report.md（含内嵌 SVG）|

## 4. Eval 执行链

`baseline.py` 与 `pipeline.py` 进程内调用 `task_config.run_eval`
（[scripts/task_config/eval_client.py](scripts/task_config/eval_client.py)，WA 专属薄 shim）。
shim 把入参原样转发给 `utils.akg_eval.eval_kernel`（[scripts/utils/akg_eval.py](scripts/utils/akg_eval.py)），
再把返回 dict 装回 `EvalResult`：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(task_dir, config, device_id, worker_urls)
     └─ utils.akg_eval.eval_kernel(task_dir, config, device_id, worker_url)
         ├─ akg_agents.core.worker.manager.register_local_worker(device_ids=[..])
         │      OR register_remote_worker(worker_url=...)         # 二选一
         ├─ akg_agents.op.verifier.KernelVerifier(...).run(...)   # verify pass
         └─ KernelVerifier.run_profile(...)                       # ref+kernel profile
```

两条路径由 `worker_url` 决定：

- `worker_url is None` → `LocalWorker` 在当前进程注册并执行
- `worker_url` 非空（`host:port` 形式）→ `RemoteWorker` 注册到 AKG worker
  manager，由 AKG 内部 RPC 与远端通信

远端 worker 的连接、排队和执行由 AKG worker 子系统接管。

Sticky baseline 语义保留：`baseline_metric` 与 `baseline_source=ref` 写定到
state.json 之后，后续轮通过 AKG verifier 内部跳过 ref profile。非有限浮点
（inf/-inf/nan）由 `utils.akg_eval._make_ok_payload` 在 dict → EvalResult 转
换前已经被 `_float()` 收成 None。

## 5. 精度容差

`/autoresearch` 每轮 verify 直接走 `akg_agents.op.verifier.KernelVerifier`，容差表来自 AKG 的 CANN MARE/MERE-aligned 分层实现。入口文件：`akg_agents/python/akg_agents/op/verifier/adapters/framework/torch.py`。

核心判定：

- 按 ref dtype 选择容差组。
- 元素级 strict / relaxed 双带判定，超 relaxed 或 outlier 比例超阈值即 FAIL。
- NaN、Inf、bool/int 走对应一致性检查。

## 6. 文件与状态布局

| 路径 | 用途 |
|---|---|
| `workspace/<op>_ref.py` / `<op>_kernel.py` | 候选 ref/kernel 输入 |
| `task.yaml` | name / arch / editable_files / ref_file / devices / eval_timeout / max_rounds / metric / `data_files` / `worker.urls` |
| `.ar_state/state.json` | **单文件控制态**：phase / owner / progress / pending_settle / 一致性 expected_* |
| `.ar_state/plan.md` | 规划与结算历史（权威态）|
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/plan_items.xml` | PLAN/DIAGNOSE/REPLAN 写给 `create_plan.py` 的 XML |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 结构化诊断报告（AGENTS.md 不变量 #10）|
| `config.yaml` | `hallucinated_scripts`（脚本名容错）+ `remote_worker.hosts`（SSH alias → repo_path / env_script）|
| `.claude/settings.json` | Hook + 权限（提交至仓库）|
| `.claude/settings.local.json` | API key / model 覆盖（不提交至 git）|

`.ar_state/` 内除 `plan_items.xml` 与 `diagnose_v<N>.md` 外均由 hook 与脚本管理，Claude 不可手写。

**Batch 目录布局**（单文件 DSL —— triton / pypto / tilelang / cpp / cuda_c / ascendc / swft）：

```
<batch_dir>/                         ← 批级
  manifest.yaml                      # prepare.py 写
  batch_progress.json                # run.py 写：每个 op 的 status / task_dir / metrics
  batch.log                          # run.py 写：claude --print 的全部 stdout
  verify_results.json                # prepare.py / verify.py 写
  refs/<op>_ref.py                   # 文件名必须严格为 <op>_ref.py
  kernels/<op>_kernel.py             # 文件名必须严格为 <op>_kernel.py

<workspace_autoresearch>/ar_tasks/<op>_<ts>_<uuid>/    ← round 级（由 /autoresearch 维护）
  kernel.py / reference.py / task.yaml
  .ar_state/{state.json, plan.md, history.jsonl, report.md}
```

**多文件 DSL（ascendc_catlass）`kernels/` 子树**：DSL adapter 标 `kernel_arg_is_directory=True` + `kernel_project_dir_name="catlass_op"` 时，[`batch/manifest.py::resolve_kernel_paths_for_op`](scripts/batch/manifest.py) 切到 per-op 子目录布局：

```
<batch_dir>/kernels/
  <op>/
    kernel.py        (或 <op>_kernel.py — 同 §2.2 的 fallback)
    catlass_op/                       # 整个项目目录，作为 /autoresearch --kernel 传入
      kernel/<>.asc
      include/<>.h
      src/catlass_torch.cpp
      CMakeLists.txt
```

manifest 解析时 case dict 同时存两个字段：`kernel`（传给 `--kernel`，单文件 DSL 是 .py，catlass 是 catlass_op 目录）+ `kernel_module`（tier-1/tier-2 import 的 Python wrapper，永远是 .py 文件）。两个字段在单文件 DSL 下指向同一路径，调用方按这两个字段读取即可。

`batch_progress.json:cases.<op>.task_dir` 连接批级与 round 级。

## 7. env.sh 契约

适用场景：tmux 非 login shell、`akg_cli worker --remote-host` 远端启动、batch `bash --login -c 'source env.sh && python ...'`。三处共用同一脚本，路径填入 `config.yaml: remote_worker.hosts.<alias>.env_script`。

**唯一职责**：source 完成后，PATH 上的 `python` 执行 `python -c "import torch_npu, triton"` 不抛异常。

模板（按本机 Python + CANN 安装方式选一）：

```bash
# 例 1：conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env 名>
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 例 2：venv
source ~/.venvs/<env 名>/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 例 3：系统 Python，仅加载 CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**写法约定**：env.sh 负责环境变量和 Python 环境初始化；工作目录由 akg_cli 切换到 `repo_path`。

验证：`source ~/env.sh && python -c "import torch_npu, triton"` 退出码 0。

## 8. akg_cli worker

Worker 是个 FastAPI HTTP daemon，跑 eval 子进程。`akg_cli worker` 子命令负责本机/远端起停、ssh -L tunnel 维护、探活。

### 启动示例

```bash
akg_cli worker --remote-host my-npu --start --backend ascend --devices 0,1
akg_cli worker --remote-host my-npu --status
akg_cli worker --remote-host my-npu --stop
```

省略 `--remote-host <alias>` 时在本机直接起 daemon；远端模式负责 SSH / tunnel。

`--start` 幂等：daemon 已在跑时返回当前状态；tunnel 抖动后再执行一次 `--start` 即可恢复。`--status` 只查询状态。

### 参数

`--start` / `--stop` / `--status` 三选一，互斥。

| flag | 类型 | 默认 | 必填 | 说明 |
|---|---|---|---|---|
| `--backend` | `ascend` / `cuda` / `cpu` | — | `--start` 必填 | 硬件后端 |
| `--devices` | csv，如 `0,1,2` | — | `--start` 必填 | device id 列表 |
| `--arch` | 字符串，如 `ascend910b3` | auto（`npu-smi info` 推断）| 可省 | 硬件 arch |
| `--port` | int | `config.yaml: worker.port`，否则 `9001` | 可省 | TCP 端口 |
| `--remote-host` | alias | — | 可省 | 走 SSH 远端模式；alias 在 `./config.yaml` 里 `remote_worker.hosts.<alias>` 定义 |

`./config.yaml: remote_worker.hosts` 字段：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path:  /abs/path/to/repo   # 必填
      env_script: /abs/path/env.sh    # 必填，source 后 PATH 上的 python 可 import torch_npu/triton
      ssh_alias:  my-npu              # 可省，默认 = key
```

`repo_path` 指远端 akg checkout。远端入口固定为 `python -m akg_agents.cli.cli worker ...`；`python` 来自 `env_script` source 后的 PATH。

## 9. Skills 库

根目录：`skills/`，按 DSL 分区：`triton-ascend/`、`triton-cuda/`、`pypto/`、`cpp/`、`cuda-c/`、`tilelang-cuda/`。每个 DSL 下进一步分为：

- `fundamentals/` — API rules、hardware constraints、basics、debugging、grid-config、memory、optimization
- `guides/` — attention、matmul、reduce、elementwise、elementwise-reduce-fused 等场景指南
- `cases/` — 单算子题型 case
- `examples/` — 端到端可运行示例
- `evolved-fix/`、`evolved-improvement/` — 故障排查与性能优化的演化知识

PLAN 阶段 hook 提示 Claude Glob `../skills/<dsl>/**/SKILL.md`、Read 1-3 个最相关的，将文件名写入 plan item rationale。

## 10. Hook 与内部机制

外部接口（slash 命令、`task.yaml`、`.ar_state/` 路径）保持稳定。修改内部实现的入口：

| 主题 | 入口 |
|---|---|
| Bash gate（命令在何 phase 合法）| [phase_policy.py](scripts/phase_machine/phase_policy.py) 头部注释 |
| Hook 接线 | [.claude/settings.json](.claude/settings.json)；脚本位于 [hooks/](scripts/hooks/)，`guard_*.py` / `post_*.py` / `stop_*.py` |
| phase 转移 | [phase_machine/state_store.py](scripts/phase_machine/state_store.py) 阶段常量；`compute_next_phase` / `compute_resume_phase` 在 [phase_policy.py](scripts/phase_machine/phase_policy.py) 末尾 |
| 测时 + verify/profile | `akg_agents.op.verifier.KernelVerifier`（在 worker 进程里跑；通过 [utils/akg_eval.py](scripts/utils/akg_eval.py) 调起）|
| Eval 执行链 | [task_config/eval_client.py](scripts/task_config/eval_client.py) `run_eval` → [utils/akg_eval.py](scripts/utils/akg_eval.py) `eval_kernel` → `akg_agents.core.worker.manager` 注册 Local/RemoteWorker → `KernelVerifier.run` + `run_profile` |
| Remote worker SSH 调度 | `akg_cli worker --remote-host` （实现在 [`cli/service/remote_dispatch.py`](../python/akg_agents/cli/service/remote_dispatch.py)，配合 sibling [`tunnel.py`](../python/akg_agents/cli/service/tunnel.py) + [`remote_probe.py`](../python/akg_agents/cli/service/remote_probe.py) + [`diagnostics.py`](../python/akg_agents/cli/service/diagnostics.py)）；config 在 WA [config.yaml](config.yaml) `remote_worker.hosts`；远端 daemon 是 [`akg_agents/worker/server.py`](../python/akg_agents/worker/server.py) |
| Triton 退化静态检查 | EDIT 前入口 [`engine/quick_check.py`](scripts/engine/quick_check.py)（docstring 列出三类退化模式）；规则实现在 `akg_agents.op.utils.code_checker.CodeChecker`，quick_check + [batch/verify.py](scripts/batch/verify.py) 各自 `CodeChecker(...).check(code)` 直接同步调（无 WA-侧 wrapper） |
| DIAGNOSE 契约 | [AGENTS.md](AGENTS.md) 不变量 #9（canonical-form bash）+ #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |

Hook 通过 [.claude/settings.json](.claude/settings.json) 接入，运行时围绕 `state.json` 维护 phase、owner 和 progress。日常使用主要通过 `/autoresearch`、`scripts/batch/*` 和 `akg_cli worker` 入口操作；内部改动按上表定位。

## 11. 并发与冲突

运行约定：

| 资源 | 推荐做法 |
|---|---|
| batch 目录 | 一个 `<batch_dir>` 同时跑一个 `run.py`，状态由 `.batch.lock` 串行化 |
| task 目录 | 一个 `<task_dir>` 同时由一个 Claude session 驱动，owner/heartbeat 写在 `state.json` |
| NPU 卡 | 一张卡交给一个本地 eval 或一个 worker；批量任务按卡分流，或统一走同一个 worker 队列 |
| worker / config | 修改 `akg_agents/`、workspace scripts 或 `config.yaml` 后重启 worker，使 daemon 读取新代码和配置 |

常用恢复入口：

| 场景 | 操作 |
|---|---|
| batch 被旧 lock 卡住 | 确认对应进程已退出后清理 `.batch.lock` |
| task 被旧 owner 卡住 | 等 `resume.heartbeat_fresh_seconds` 过期，或确认旧 session 已退出后清理 `state.owner` |
| tunnel 不通 | 先 `akg_cli worker --remote-host ... --status`，再按需 `--start` |
| worker 返回旧源码或旧配置行为 | 重启 worker |
