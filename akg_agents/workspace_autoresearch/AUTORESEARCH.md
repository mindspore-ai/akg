# workspace_autoresearch

Claude Code 驱动的算子自动优化框架。本目录是 `akg_agents/` 的使用态工作空间：
phase machine + hooks + slash command 在 `scripts/`，verifier 与 worker 走
`utils.akg_eval` 桥（直接 import `akg_agents.op.verifier.KernelVerifier` +
`akg_agents.core.worker.manager`，不 vendor）。

**主循环**：scaffold → BASELINE → PLAN → EDIT → eval → KEEP/DISCARD → 直到 `max_rounds` → FINISH。

第一次用：按下表选一条「操作」路径走完即可，技术细节都在「参考」里按主题归档。

| 我要做 | 跳转 |
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
| 远程 worker daemon 协议 / 版本漂移陷阱 | [参考 §8](#8-远程-worker-内部) |
| Triton 调优 markdown 库 | [参考 §9](#9-skills-库) |
| 修内部代码前先看 | [参考 §10](#10-hook-与内部机制) |
| 并发跑 / 多 session / 共用卡的注意事项 | [参考 §11](#11-并发与冲突) |

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

`workspace/<op>_ref.py`：PyTorch 标准答案，暴露 `class Model(nn.Module)` + `get_inputs()`（或 `get_input_groups()` 多 shape）+ `get_init_inputs()`。

`workspace/<op>_kernel.py`：种子 kernel，暴露 `class ModelNew(nn.Module)`，必须含 `@triton.jit` 并实际 launch（否则 [`engine/quick_check.py`](scripts/engine/quick_check.py) 拒绝）。

最小可跑示例（relu，仅供首次跑通参考）：

<details><summary>relu_ref.py / relu_kernel.py 示例</summary>

```python
# workspace/relu_ref.py
import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x): return F.relu(x)
def get_inputs():      return [torch.randn(1024, 1024, dtype=torch.float16)]
def get_init_inputs(): return []
```

```python
# workspace/relu_kernel.py
import torch, torch.nn as nn, triton, triton.language as tl

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        if not x.is_npu: x = x.npu()
        x = x.contiguous()
        out = torch.empty_like(x)
        n, BLOCK = x.numel(), 1024
        _relu_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
        return out
```
</details>

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
| 远端 NPU 机 | (1) `git clone https://atomgit.com/mindspore/akg.git` 到 `<repo_path>`（默认目录名 `akg/`；B.2 会把这条路径写进 config）。(2) 写 `~/env.sh`：source 完毕后 `python -c "import torch_npu, triton"` 不抛异常，模板见 [§7](#7-envsh-契约)。(3) `npu-smi info` 能列出设备。**无需** Claude Code CLI（远端只跑 worker，不跑 claude）。 |
| 本机 | (1) 装 Python ≥ 3.10、PyYAML、Claude Code CLI。(2) **无需** torch_npu / CANN / NPU 硬件（eval 全部走远端）。(3) `~/.ssh/config` 配好 alias（下文以 `my-npu` 为例）+ 密钥免密登录远端。 |

<details><summary><code>~/.ssh/config</code> 怎么配 + 跑前自检</summary>

**第一步：本机 `~/.ssh/config` 给远端机起个别名**。别名是为了让后续命令短一点（不用每次敲 IP + 用户名），叫什么都行，下面用 `my-npu`。

```text
# 文件路径：本机 ~/.ssh/config（没有就新建，注意是 config 不是 config.txt）
Host my-npu                            # 这是别名，自己看的，叫什么都行
    HostName 192.168.x.x               # 远端机真实 IP 或域名
    User <remote-user>                 # 登录远端用的账号，比如 root
    # 私钥若在本机 ~/.ssh/ 下的默认位置（id_rsa / id_ed25519 / id_ecdsa 任一），下一行不用写；
    # 私钥放在本机别处才需要补这行，路径换成本机实际位置：
    # IdentityFile <本机私钥绝对路径>
    # 如果中间要过跳板机/堡垒机：
    # ProxyJump bastion
```

> 如果本机之前从没生成过 ssh 密钥，可以走最简流程（本机和远端都各有自己的 `~/.ssh/`，下面每一步都标了在哪台机器跑、读/写哪一边的文件）：
>
> 1. **本机** 命令行跑 `ssh-keygen`（一路回车用默认即可），在 **本机** `~/.ssh/` 下生成两个文件——`id_rsa` 是**私钥**（留本机，永远不要拷给远端、不要发出去），`id_rsa.pub` 是**公钥**（接下来要送到远端的那一份）。
> 2. **本机** 跑 `ssh-copy-id my-npu`。这条命令做的事情：用密码方式登一次远端，把 **本机** `~/.ssh/id_rsa.pub` 的内容追加到 **远端** `~/.ssh/authorized_keys` 末尾——远端的 ssh 服务一旦在 `authorized_keys` 里看到本机公钥，之后本机用对应私钥连上来时就不再要密码。前提是远端的 ssh 密码登录暂时是开的（系统默认通常开着），运行时会让一次性输入远端账号的密码。
> 3. 配完之后 **本机** 跑 `ssh my-npu` 不再问密码，就算成功。

> **注意**：上面 `Host` 后面的别名必须和 B.2 里 `remote_worker.hosts:` 下面写的 key 完全一样（autoresearch 默认按 key 当 ssh 目标用）。若 `~/.ssh/config` 里 `Host` 名跟 B.2 的 key 不一样，B.2 里加一行 `ssh_alias: <ssh 实际用的 Host 名>` 覆盖即可。

**第二步：跑前自检**。在本机命令行敲下面这一行，确认三件事一次性都过：免密 ssh 通、远端 env 完好、NPU 看得见。三个都过了才上 B.2 / B.3。

```bash
# 命令在本机敲。ssh 后面单引号里那一串都是在远端执行的，所以路径里的 <远端账号> 指远端机
# 的用户名，env.sh 是远端机上的脚本。
ssh my-npu 'source /home/<远端账号>/env.sh \
    && python -c "import torch_npu, triton" \
    && npu-smi info | head -3'
```

**预期看到**（按顺序）：
1. 没有 `Permission denied` / `Connection refused` —— ssh 通了
2. `python -c ...` 一行不报错（特别是没有 `ModuleNotFoundError`）—— 远端 conda 环境和 CANN 都装好了
3. 打印一段类似 `npu-smi 25.x.rcN  Version: ...` 的表头 —— NPU 驱动和卡都在

**任一步出错就在这里修掉，别先去跑 akg_cli**（akg_cli 启动失败时同样的错误会被埋进远端 `/tmp/akg_worker_9111.log` 里，定位起来麻烦得多）。常见错和对应处理：

| 报错关键字 | 大概率原因 | 怎么修 |
|---|---|---|
| `Permission denied (publickey)` | 公钥没进远端 `~/.ssh/authorized_keys`，或本机私钥路径填错 | `ssh-copy-id my-npu` 把公钥推过去；或检查 `IdentityFile` 路径 |
| `Connection timed out` / `No route to host` | 远端 IP 写错，或者本机当前网络连不到 | 改 `HostName`；或确认本机已接入能访问远端的网络（VPN / 内网） |
| `ModuleNotFoundError: torch_npu` | 远端 env.sh 路径错，或远端 conda 环境没装好 | ssh 进远端手动 `source <远端 env.sh 绝对路径> && python -c "import torch_npu"`，调通了再回来跑本机这条自检命令 |
| `npu-smi: command not found` | 远端 CANN 没装，或没在远端 `PATH` 里 | 远端 env.sh 里补 CANN 的 `source set_env.sh`，或在远端装 CANN |

</details>

### B.2 [本机] 配 worker host

`config.yaml`：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path:  /home/<user>/akg               # 远端 akg checkout 路径（必填）
      env_script: /home/<user>/env.sh            # 必填，source 后 import torch_npu 不挂
      python:     python                         # 可选
      ssh_alias:  my-npu                         # 可选，默认 = key
```

字段语义见 [参考 §8](#8-远程-worker-内部)。`akg_cli` 默认读 `./config.yaml`；放别处用 `--remote-config <path>` 覆盖。

### B.3 [本机] 启动远端 worker（akg_cli 自动开 SSH tunnel）

```bash
akg_cli worker --remote-host my-npu --start \
    --backend ascend --arch ascend910b3 --devices <NPU-id> --port 9111
```

一条命令做两件事：(1) SSH 到远端起 daemon（直接 spawn local checkout 的 `python -m akg_agents.worker.server`，PYTHONPATH 锁到 `repo_path/akg_agents/python` —— 避开 pip install 那份的版本漂移，见 [§8](#8-远程-worker-内部)），(2) 本机起 `ssh -L 9111:127.0.0.1:9111`，tunnel pid 存 `~/.akg_agents/tunnels/9111.pid`。后续 `--worker-url 127.0.0.1:9111` 透传。

多租户机器：9111 经常被别人占，挑 9112+ 更稳（同步改 `--port` 与下游 `--worker-url`）。

验证：

```bash
akg_cli worker --remote-host my-npu --status --port 9111
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

先 SIGTERM 本机 tunnel（按 `~/.akg_agents/tunnels/9111.pid`），再 SSH `lsof -ti :9111 | xargs -r kill` 杀远端 daemon。`lsof -ti` 只取占该端口的 PID，避免误杀别人。

---

## C. 批量

`scripts/batch/` 对一批 `(ref.py, kernel.py)` 任务循环跑 `/autoresearch`。批级状态写 `<batch_dir>/batch_progress.json`，round 级仍在 `ar_tasks/`。

### C.1 标准流程

约束：cwd 必须在 `autoresearch/` 内；长跑用 tmux/screen 包；`--devices` 必填，整批串行复用一张卡（除非用 `--worker-url`）。

```bash
BATCH_DIR=/tmp/batch_001
DEVICE=0

# 0. 进入子目录
cd workspace_autoresearch

# 1. 放 ref/kernel（命名必须严格 <op>_ref.py / <op>_kernel.py，见参考 §2）
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp workspace/*_ref.py    $BATCH_DIR/refs/
cp workspace/*_kernel.py $BATCH_DIR/kernels/

# 2. Tier-1 预检：纯静态（语法 / import / 必备 export），任意机器可跑、不需要 NPU 或 torch_npu
python scripts/batch/prepare.py $BATCH_DIR

# 3.（可选）Tier-2 预检：在本进程里真跑 ref vs kernel 比对输出
#    要求当前机器有 NPU + torch_npu（直接 import 进程内执行，不走 worker），
#    所以双机配置下不能在本机跑，要 SSH 到 NPU 机执行；用本机直跑的方案省略本步。
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

### C.2 配合远端 worker 跑批（本机无 NPU）

如果本机不是 NPU 机（即 B 章场景），批跑改动很小，整体仍按 [C.1](#c1-标准流程) 操作，但有两处差异：

1. **先按 [B.3](#b3-本机-启动远端-workerakg_cli-自动开-ssh-tunnel) 启动远端 worker 并建立本机 tunnel**（每次开新 batch 前确认 tunnel 还活着：`akg_cli worker --remote-host my-npu --status --port 9111`）。
2. **`run.py` 加 `--worker-url 127.0.0.1:9111`**，且本机 tmux 不需要 `bash --login + source env.sh`（本机没有 torch_npu / CANN，eval 全走远端 worker）。

第 4 步的 tmux 命令变为：

```bash
tmux new -d -s ar_batch \
    "python -u scripts/batch/run.py $BATCH_DIR \
       --worker-url 127.0.0.1:9111 --devices $REMOTE_DEVICE \
       2>&1 | tee -a $BATCH_DIR/batch.log"
```

`--devices` 在有 `--worker-url` 时仍要给，但取值是**远端**卡下标（透传到每个 op 的 task.yaml）。

其它步骤（prepare / monitor / summarize）和 [C.1](#c1-标准流程) 一样在本机直接跑；**唯一例外是 Tier-2 verify**（step 3，`verify.py --full`）：它在进程内直接 `import torch_npu` 跑 ref/kernel，**不走 worker**，所以本机没有 NPU 跑不了。需要做完整预检的话，得 SSH 到 NPU 机执行 `python scripts/batch/verify.py <batch_dir> --full`，或者跳过 Tier-2 直接让 `/autoresearch` 在每轮 eval 时自带 verify。

### C.3 监控工具一览

| 工具 | 数据源 | 用途 |
|---|---|---|
| `monitor.py` | progress + ar_tasks/ | 实时队列 + phase + heartbeat |
| `monitor.py --dashboard` | 同上 | `execvp` 至 `dashboard.py` 看当前 task TUI |
| `summarize.py` | 仅 progress JSON | 离线汇总，复制粘贴友好 |
| `tail -f batch.log` | claude stdout | 查 hook 输出、Edit、Bash |

### C.4 断点续跑 / 重试

- `done` 不再跑
- `error` 默认跳过；`--retry-errored` 重新纳入
- `pending` 自动续
- `running`（终止瞬间在跑的 op）下次启动时自动降级为 `error`，note 记 `stale running, demoted on batch restart`
- **transient 自动续**（同 batch 内）：单个 op 跑到一半 `claude --print` 进程异常退（rc≠0，常见 ECONNRESET / Stream idle timeout / 其他 transient API 错），但 framework state 仍 intact（`progress_initialized=True`，phase 不是 FINISH）时，supervisor 会自动 `/autoresearch --resume <task_dir> --force` 接续，最多 `batch.transient_retries` 次（默认 3，配 [`config.yaml`](config.yaml)）。接续成功的 op `batch_progress.json::cases.<op>.note` 会记 `transient_retries=N`，区分一遍过 vs 靠 wrapper 救活。`rc=0 + phase != FINISH`（LLM 自然 stop）或 `rc != 0 + no progress`（baseline 没 commit）属于真失败，不 retry，直接记 `error` 跳下一个 op。

> 同一 batch_dir **禁止并发** 多个 `run.py`（`.batch.lock` 排它）。死进程残留的 lock 下次启动自动判活清理。

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

操作部分已经涵盖正常使用。下面是按主题归档的细节，平时不用读，碰到问题时按目录跳转。

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

注：`arch`（如 `ascend910b3`）由 `--devices` 选中的卡经 `npu-smi info` 自动推断，写进 `task.yaml` 仅供 dashboard / report 显示，不需要用户传。

---

### 1.2 `akg_cli worker` （worker 启停 + tunnel）

AKG canonical CLI。三种模式互斥：

| 模式 | 用法 |
|---|---|
| `--start` | 启动 worker daemon。如配合 `--remote-host`，会先 SSH 到远端 spawn `python -m akg_agents.worker.server`（PYTHONPATH 锁到 `repo_path/akg_agents/python`，避开 pip install 那份），再在本机起 `ssh -L <port>:127.0.0.1:<port>` tunnel |
| `--stop` | 停 `--port` 上的 daemon；配合 `--remote-host` 时先 SIGTERM 本机 tunnel，再 SSH `lsof -ti :<port> \| xargs kill` 杀远端 |
| `--status` | `curl 127.0.0.1:<port>/api/v1/status`；配合 `--remote-host` 走的是本机 tunnel（假定 `--start` 已建好）|

主要 flag：

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--start` / `--stop` / `--status` | flag | ✅ 三选一 | — | 互斥 |
| `--backend` | `ascend` / `cuda` | `--start` 时必填 | — | 硬件后端 |
| `--arch` | 字符串（如 `ascend910b3`） | ❌ | `a100` | 写入 worker 自报的 arch；远端启动可显式传以覆盖 |
| `--devices` | 逗号分隔 | ❌ | `0` | worker 管理的卡集合 |
| `--port` | int | ❌ | 9001 | TCP 端口；WA 习惯用 9111+ |
| `--host` | IP | ❌ | `0.0.0.0`（本机）/ `127.0.0.1`（`--remote-host`） | 绑定地址。远端模式强制 loopback——worker 只经 SSH tunnel 访问 |
| `--remote-host` | SSH alias | ❌ | (本机) | 通过 SSH 在 `remote_worker.hosts.<alias>` 定义的远端机执行；`--start` 额外打通本机 tunnel |
| `--remote-config` | path | ❌ | `./config.yaml` | 读 `remote_worker.hosts` 的 yaml 路径 |

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

违反将导致流程无法启动。

| 项 | 约束 |
|---|---|
| ref 文件名 | 单跑任意；批跑严格 `<op>_ref.py` |
| kernel 文件名 | 单跑任意；批跑严格 `<op>_kernel.py` |
| ref 暴露 | `class Model(nn.Module)` + `get_init_inputs()`，并二选一：`get_inputs()` 单 shape / `get_input_groups()` 多 shape |
| kernel 暴露 | `class ModelNew(nn.Module)`，含 `@triton.jit` 且实际 launch；forward 不得用 `torch.*` / `F.*` / tensor 方法 / `@` 算子 / Python 循环完成计算。规则由 `akg_agents.op.utils.code_checker.CodeChecker` 强制（syntax / py_compile / import / DSL 合规 / autotune / forbidden torch.* in forward），[`engine/quick_check.py`](scripts/engine/quick_check.py) 和 [`batch/verify.py`](scripts/batch/verify.py) 直接 `CodeChecker(...).check(code)` 调起 |
| 同目录数据文件 | ref 通常要从同目录读取 shape 列表 / 缓存（`.json`/`.pt`/`.npz`），scaffold 按这条规则决定哪些一起打包进 task_dir：去掉 ref 文件名的 `.py` 后缀得到一个名字，**同名**的、或者以**它加下划线开头**的同目录 `.json`/`.pt`/`.npz` 会被拷进去；其余的 scaffold 看不见。<br/>例：ref `31_IOU.py` 可配 `31_IOU.json`（同名）或 `31_IOU_cases.json`（加下划线前缀）；但 ref 改叫 `iou_ref.py` 后，原来那个 `31_IOU.json` 就不再匹配，要么把 ref 改回 `31_IOU.py`，要么把数据文件改成 `iou_ref.json`（scaffold 不会自动去掉 `_ref` 后缀帮你认别名）。<br/>没匹配上的数据文件不会进 task_dir，eval 时报 `FileNotFoundError`。NPUKernelBench 原生 `<index>_<CamelCase>.py` + `<index>_<CamelCase>.json` 直接可用。 |
| task.yaml 字段 | fixed schema（[loader.py](scripts/task_config/loader.py)）；拼写错误会被默认值覆盖 |
| task_dir 命名 | scaffold 生成的格式：`ar_tasks/<op_name>_<unix_ts>_<uuid6>/`。三段分别承担：op_name 让人读得懂、unix_ts（`int(time.time())`）保证时间排序、uuid6（`uuid.uuid4().hex[:6]`）防同秒撞名。batch/manifest、resume、dashboard 都按这三段反解，手动 rename 会让它们找不到任务。 |

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

WA 没有 vendor CA 的 `scripts/eval/` 包 / `scripts/worker/server.py` /
`utils/eval_runner.py` / `engine/eval_kernel.py` / `task_config/package_builder.py`，
也没有 CA 的 HTTP `/api/v1/run` 多部分上传通道。两条路径的差别全部塌缩到
`worker_url` 是否非空：

- `worker_url is None` → `LocalWorker` 在当前进程注册并执行
- `worker_url` 非空（`host:port` 形式）→ `RemoteWorker` 注册到 AKG worker
  manager，由 AKG 自己的 RPC 与远端通信

远端 worker 协议、tar.gz 打包、disconnect watch、HTTP 499 等 CA 独有的传输细节
统统由 AKG worker 子系统接管，WA 不重复实现。

Sticky baseline 语义保留：`baseline_metric` 与 `baseline_source=ref` 写定到
state.json 之后，后续轮通过 AKG verifier 内部跳过 ref profile。非有限浮点
（inf/-inf/nan）由 `utils.akg_eval._make_ok_payload` 在 dict → EvalResult 转
换前已经被 `_float()` 收成 None。

## 5. 精度容差

`/autoresearch` 每轮 verify 直接走 `akg_agents.op.verifier.KernelVerifier`：
容差表是 AKG 自己的（CANN MARE/MERE-aligned 分层），细节见
`akg_agents/python/akg_agents/op/verifier/adapters/framework/torch.py`。WA 不再
本地维护一份并列表格，避免与 AKG 漂移。

- 按 ref dtype 取 `(rtol, atol, outlier_rtol, outlier_atol, outlier_ratio)`：fp32 → `(1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)`、fp16 → `(9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005)`、bf16 → `(7.81e-3, 1e-2, 7.81e-2, 1e-1, 0.010)`，未知 dtype 回落 fp32（值由 AKG 侧 owns，列在此仅作参考）
- 元素级双带判定：`strict_tol = atol + rtol·|ref|`，`relaxed_tol = outlier_atol + outlier_rtol·|ref|`；`hard_fail`（超 relaxed）> 0 即 FAIL，或 `outlier`（在 strict 与 relaxed 之间）超 `outlier_ratio · total` 即 FAIL
- 额外硬性检查：NaN 位置一致、Inf 位置 + 符号一致、bool/int 精确匹配

WA 侧 [`scripts/utils/correctness.py`](scripts/utils/correctness.py) 仅是 no-op
stub，给 `batch/verify.py` 的 pre-batch 数值审计留接口；真正的 verify 还在 AKG
worker 内部跑，pre-batch 审计如要启用，得在 stub 里调 KernelVerifier 而不是
重写一遍对比逻辑。

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

**Batch 目录布局**：

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

`batch_progress.json:cases.<op>.task_dir` 连接批级与 round 级。

## 7. env.sh 契约

适用场景：tmux 非 login shell、`akg_cli worker --remote-host` 远端启动、batch `bash --login -c 'source env.sh && python ...'`。三处共用同一脚本，路径填入 `config.yaml: remote_worker.hosts.<alias>.env_script`。

**唯一职责**：source 完成后，`python -c "import torch_npu, triton"` 不抛异常。

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

**禁项**：env.sh 中不得使用 `cd`（外部 akg_cli 已 `cd repo_path`）、`exec`、`set -e`。

验证：`source ~/env.sh && python -c "import torch_npu, triton"` 退出码 0。

## 8. 远程 worker 内部

WA 跑 `--worker-url` 路径时，eval 顺这条链走：

```
utils.akg_eval.eval_kernel
 ├─ register_remote_worker(worker_url=http://host:port)   # bridge 自动补 http://
 │   └─ 内部 GET /api/v1/status 探活 + 取 capacity
 ├─ wm.select() → RemoteWorker 实例
 └─ KernelVerifier.run / .run_profile
     ├─ RemoteWorker.acquire_device   → POST /api/v1/acquire_device
     ├─ RemoteWorker.verify           → POST /api/v1/verify
     ├─ RemoteWorker.profile          → POST /api/v1/profile
     └─ RemoteWorker.release_device   → POST /api/v1/release_device
```

Daemon 是 [`akg_agents/worker/server.py`](../python/akg_agents/worker/server.py)（FastAPI + uvicorn），暴露 8 个 `/api/v1/*` endpoint：`status` / `verify` / `profile` / `acquire_device` / `release_device` / `generate_reference` / `profile_single_task` / `docs/{doc_name}`。Bridge 端读 `/status` 自动补 `task.yaml` 缺的 `arch` 字段（scaffold 在远端模式下故意不填，因为 orchestrator 可能没 NPU 探测）。

### 为啥不是 CA 单 endpoint

CA 上游 `scripts/worker/server.py` 用单一 `/api/v1/run` 收 tar.gz 一次性 verify+profile。AKG 这边 stage endpoint 是有意为之——`evolve.py` / `adaptive_search` / langgraph workflow 也调 AKG worker，它们需要跨 stage 复用同一个 acquire 的 device（同 kernel 跑多 shape profile，复用 JIT cache）。把 AKG 砍成 CA 单 endpoint 会拖崩这几个 caller。

### 版本漂移陷阱（akg_cli 远端模式如何回避）

不带 `--remote-host` 的 `akg_cli worker --start` 走的是 **当前 shell 里 import 的 akg_agents**——pip install 的版本如果落后于 checkout，daemon 就可能少 endpoint，客户端 `acquire_device` 会撞 HTTP 404 `Not Found`。

`akg_cli worker --remote-host <alias> --start` 通过 [§B.3](#b3-本机-启动远端-workerakg_cli-自动开-ssh-tunnel) 描述的命令，**主动**把远端 PYTHONPATH 锁到 `repo_path/akg_agents/python`，spawn `python -m akg_agents.worker.server`，让 daemon 跑 checkout 那份代码，而不是 pip install 那份。

要在本机直跑（同机调试）又想避开漂移，自己复刻同样的 PYTHONPATH 锁定即可：

```bash
PYTHONPATH=<repo>/akg_agents/python:$PYTHONPATH \
WORKER_BACKEND=ascend WORKER_ARCH=ascend910b3 \
WORKER_DEVICES=0 WORKER_HOST=127.0.0.1 WORKER_PORT=9111 \
nohup python -m akg_agents.worker.server > /tmp/akg_worker_9111.log 2>&1 &
```

### Follow-up

AKG worker 加 `/api/v1/run` aggregate endpoint：WA bridge 走 aggregate 把客户端协议面缩到 `/status` + `/run` 两条，消除版本漂移敏感面；同时 stage endpoint 留给 AKG 其它 caller 复用 device。WA 这边把 `utils/akg_eval.py` 改成发 `/run` 一锅端，省掉 `acquire_device → verify → profile → release_device` 四次 round-trip 的延迟。今天没做。

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
| Remote worker SSH 调度 | `akg_cli worker --remote-host` （实现在 [`cli/service/worker_remote.py`](../python/akg_agents/cli/service/worker_remote.py)）；config 在 WA [config.yaml](config.yaml) `remote_worker.hosts`；远端 daemon 是 [`akg_agents/worker/server.py`](../python/akg_agents/worker/server.py) |
| Triton 退化静态检查 | EDIT 前入口 [`engine/quick_check.py`](scripts/engine/quick_check.py)（docstring 列出三类退化模式）；规则实现在 `akg_agents.op.utils.code_checker.CodeChecker`，quick_check + [batch/verify.py](scripts/batch/verify.py) 各自 `CodeChecker(...).check(code)` 直接同步调（无 WA-侧 wrapper） |
| DIAGNOSE 契约 | [AGENTS.md](AGENTS.md) 不变量 #9（canonical-form bash）+ #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |

### Hook 接线概要

控制态单文件：`<task_dir>/.ar_state/state.json`（phase / owner / progress / pending_settle / 一致性 expected_*）。任务的归属判定靠 `state.owner.session_id` 与环境变量 `CLAUDE_CODE_SESSION_ID` 相等；当前会话所驱动的任务由 `phase_machine.find_active_task_dir()` 通过扫描 `ar_tasks/` 解析得到。

```
hook_guard_bash (bash 之前)
  从 state.json 读 phase，按 phase 表允许/拒绝 AR 脚本
  仅认裸 scripts/...（或 ./scripts/）为 AR 调用；cwd = autoresearch/
  带前缀的 scripts/... 既被判非规范、磁盘上也不存在
  blessed CLIs：pipeline / baseline / create_plan / parse_args / scaffold / resume / dashboard
  其余 python scripts/*.py 走 _LIBRARY_NOT_CLI（pointed hint）或 generic "Unknown script" 拒绝

hook_post_bash (任意 bash 之后)
  检测 AR_TASK_DIR=    → set_task_dir 写 state.owner，首次激活 PhaseController.on_activation_ready → BASELINE
  检测 baseline.py     → workflow.run_baseline_init 已在子进程推进 phase
  检测 pipeline.py     → pipeline 自身已写 phase；hook 仅 echo
  检测 create_plan.py  → on_plan_validated → EDIT

hook_post_edit (Write/Edit kernel.py 之后)
  EDIT phase 下编辑 kernel.py → 提示运行 pipeline.py

hook_post_task (任意 Task 工具之后)
  bump state.last_touched（防长跑 DIAGNOSE 子代理超 heartbeat 窗口）
  DIAGNOSE artifact 校验

hook_stop_save (Claude 试图 Stop 时)
  FINISH → 允许
  BASELINE 且无 progress（baseline 提交闸把它挡下来的"baseline 待补"状态）→ 允许
  其余 phase → 拒绝，要求继续主循环
```

## 11. 并发与冲突

下面列出几种典型并发场景：哪些框架已经挡住、哪些需要你自己避开、撞到了怎么排错。

### 11.1 框架已挡住的（你不用担心）

| 场景 | 保护机制 | 行为 |
|---|---|---|
| 同一 `<batch_dir>` 两个 `run.py` 同时启动 | `<batch_dir>/.batch.lock`，`O_CREAT\|O_EXCL` 原子创建 | 后启动的立即报错退出。死进程残留的 stale lock 在下次启动时被判活清理 |
| 同一 `<task_dir>` 两个 claude session 同时 attach | `state.owner.session_id` + heartbeat 新鲜度窗口（`config.yaml: resume.heartbeat_fresh_seconds`，默认 180s） | 后到的 session 在 `set_task_dir` 处被 refuse，stderr 提示对方的 session_id |
| 用 `/autoresearch --resume` 续一个还在跑的 task | 同上 | 同上拒绝 |
| 写 `state.json` 中途崩溃 | tmp 文件 + `os.replace` 原子替换 | state.json 永远是上一稳定版或新版，不会读到撕裂内容 |
| 启动第二个 worker 占同一端口 | 内核 bind 失败 | uvicorn 直接报 `address already in use` 退出，daemon 起不来 |

### 11.2 框架挡不住的（必须你自己避免）

| 场景 | 后果 | 怎么避 |
|---|---|---|
| 同一张 NPU 卡同时跑两份 eval（local-eval × 2 / local-eval + worker / 同卡两个 worker） | **没有跨进程的卡级互斥**。会撞 device，症状可能是 NPU OOM、kernel crash、profile 时延异常、甚至触发 ECC | 一卡一用：要么这张卡只给一个 worker（其他人走 `--worker-url 127.0.0.1:9111` 排队），要么这张卡只给一个 `/autoresearch` |
| 两个 batch（不同 batch_dir）但共用同一张 `--devices` | 同上，`.batch.lock` 只防同 batch_dir 并发，跨 batch 不互斥 | 不同 batch 拨不同卡，或都走同一个 worker（worker 内部对自身管理的卡做了 queue 串行化） |
| 一台机器既跑本地 `/autoresearch --devices N`，又给 N 上跑了 worker | 同上 | 二选一：要么 worker 占着这张卡、本机其他人都走 `--worker-url`；要么不开 worker、本机直接 local-eval |
| 手动改 `state.json` / 删 `history.jsonl` 中间几行 | 一致性 gate 在下次 hook 触发时会报 `state inconsistent`，task 卡住 | 不要手编辑这两个文件。要重来直接删整个 task_dir |
| Worker 启动期间改了 `akg_agents/` 或 `config.yaml` | daemon 不重读源码 / config；下次 verify 仍跑旧版 | 重启 worker：`akg_cli worker --remote-host my-npu --stop --port 9111` 再 `--start ...` |

### 11.3 常见症状对应到这里

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `[batch] error: another run.py is active (.batch.lock held by pid N)` | 11.1 第 1 行 | 如果对方真在跑，等；不然手动 `rm .batch.lock` |
| `[state_store] WARNING: refusing to claim <td> — owned by session_id=...` | 11.1 第 2 行（或对方 session 没正常退出，heartbeat 还在新鲜窗口内） | 等 `resume.heartbeat_fresh_seconds` 过期；或确认对方 session 已死后手动清 `state.owner` |
| eval 输出指标抖动、对照 baseline 时延变 2-3 倍 | 11.2 同卡多用户 | 检查同卡有没有别的 `claude --print` / worker 进程在跑 |
| `Worker tunnel 127.0.0.1:9111 unreachable` | tunnel 进程死了 / 远端 daemon 死了 / 端口被别的进程占了 | `akg_cli worker --remote-host my-npu --status --port 9111` 确认；不行就 `--stop` 再 `--start` |
| `/run` 返回 HTTP 503，提示某些 `.py` 改了 | 11.2 worker 源码漂移闸 | 重启 worker |
