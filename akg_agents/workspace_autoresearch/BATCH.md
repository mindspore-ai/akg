# 批量跑 `/autoresearch`

对一批 `(ref.py, kernel.py)` 全自动跑 `/autoresearch`，每个 op 一个 headless
`claude --print` session。脚本在 [.autoresearch/scripts/batch/](.autoresearch/scripts/batch/)。

**适用场景**：10 个以上 op、想下班后无人值守跑、想多 op 复用同一 worker daemon。
单 op debug 直接进 `claude` 交互模式粘 `/autoresearch` 就行。

---

## Quick Start

```bash
# ── 0. 一次性约定 ─────────────────────────────────────────────────────
BATCH_DIR=/path/to/my_batch    # 装这一批输入 + 批级状态的目录（不必预创建）
cd /path/to/claude-autoresearch   # 所有相对路径都基于 repo 根

# ── 1. 摆 ref/kernel 文件（命名约定强制：<op>_ref.py / <op>_kernel.py）
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp my_ops/*_ref.py    $BATCH_DIR/refs/
cp my_ops/*_kernel.py $BATCH_DIR/kernels/

# ── 2. discover + 静态预检（语法 / import / 必备 export）
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --dsl triton_ascend

# ── 3. 起 worker daemon（akg 主体的 worker；持久占卡，所有 op 串行提交）
akg_cli worker --start --backend ascend --arch <your-ascend-arch> --devices 0 --port 9111
curl -s --noproxy '*' http://127.0.0.1:9111/api/v1/status

# ── 4. 后台跑全批（tmux 持久化，SSH 断了也不挂）
tmux new -d -s ar_batch \
  "python -u .autoresearch/scripts/batch/run.py $BATCH_DIR --worker-url 127.0.0.1:9111"

# ── 5. 另开终端监控
python .autoresearch/scripts/batch/monitor.py $BATCH_DIR

# ── 6. 跑完汇总
python .autoresearch/scripts/batch/summarize.py $BATCH_DIR
```

每个 op 顺序执行，默认 `--max-rounds 30`，单 op 约 30-60 分钟。`run.py` 末尾会
打印"下一步该跑啥"（retry errored / resume pending），不用回手册查。

> ⚠️ **必须用 tmux 不能用裸 `nohup`**：SSH 一关 SIGHUP 会把子进程的 `claude`
> 进程逐个干掉。tmux server 是常驻 daemon，batch 树挂它下面跟 ssh session
> 解耦。等价的纯命令行：`setsid nohup python -u ... < /dev/null > batch.log 2>&1 &`。

---

## 各步骤详解

### Step 1 — 摆 ref/kernel 文件

文件名**严格遵守**：`<op_name>_ref.py` / `<op_name>_kernel.py`。`prepare.py` 按
相同 `op_name` 自动配对；只有单边的 op 会以 warning 打到 stderr，不进 manifest。

`op_name` 自由选定，只要左右两边一致即可。

### Step 2 — `prepare.py`：discover + 静态预检

```bash
# 第一次（必须传 --dsl）
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --dsl triton_ascend

# 加 / 删了 ref/kernel 文件后重新同步（沿用 manifest 里已有的 dsl/dirs）
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR

# 筛选子集
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --filter '*norm'
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --exclude 'foo*'   # 可重复

# 只 discover 不 verify
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --skip-verify
```

静态检查（每个 op 独立 subprocess，秒级）：

1. ref / kernel 文件 Python 语法编译过
2. import 模块能成功（缺依赖立即暴露）
3. 必备 export：`ref.py` 有 `Model` / `get_inputs` / `get_init_inputs`；`kernel.py` 有 `ModelNew`

输出表格 + `$BATCH_DIR/verify_results.json`。退出码 0=全过 / 1=有 fail/error。

正确性的完整数值验证走 `/autoresearch` 实跑时的 akg KernelVerifier，pre-flight 只
负责截住 syntax / import / 缺符号这类一秒钟就能查的硬错误。

### Step 3 — Worker daemon

worker 直接用 akg 主体的 `akg_cli`（worker 端 `pip install -e akg_agents/` 即可）：

```bash
akg_cli worker --start --backend ascend --arch <your-ascend-arch> --devices 0 --port 9111
akg_cli worker --stop
```

挑卡：`npu-smi info` / `nvidia-smi` 看 HBM 占用，闲卡编号传给 `--devices`。

> 不传 `--devices` 也不传 `--worker-url` 时，`run.py` 默认 `--worker-url 127.0.0.1:9111`
> 并启动时强制 health check，daemon 没起会立即报错并打印怎么起，不会埋在第一个
> op 几千行日志里才炸。

### Step 4 — `run.py`：跑全批

```bash
# 标准用法：worker daemon
tmux new -d -s ar_batch \
  "python -u .autoresearch/scripts/batch/run.py $BATCH_DIR --worker-url 127.0.0.1:9111"

# 本地 in-process eval（不用 daemon；多 op 同卡可能抢资源 hang，不推荐）
python .autoresearch/scripts/batch/run.py $BATCH_DIR --devices 0
```

每个 op 在独立 headless `claude --print` 里跑完整 `/autoresearch` 流程：

```text
for each pending op:
  起 claude --print  (cwd = repo 根)
    └→ Claude 跑 /autoresearch → scaffold → export AKG_AGENTS_AR_TASK_DIR
       → BASELINE → PLAN → EDIT 循环 → FINISH
    └→ stdout 实时 stream 到 batch.log
  claude 退出后 host 读 .ar_state/.phase 决定 done / error
  写 batch_progress.json
  下一个 op
```

`run.py` 跑完会打印总结 + retry/resume 命令。

### Step 5 — 监控

| 工具 | 用途 |
|------|------|
| `monitor.py $BATCH_DIR` | 默认 watch loop（15s 刷新，`-n` 调）。看队列 / 当前 op / phase / metrics |
| `monitor.py $BATCH_DIR --dashboard` | `execvp` 进 active task 的全 TUI dashboard |
| `tail -f $BATCH_DIR/batch.log` | Claude 实时输出（`[AR Phase: ...]` / Edit / Bash 全看到） |
| `tmux attach -t ar_batch` | 进 tmux 看屏幕（`Ctrl-b d` 脱离） |
| `tmux ls \| grep ar_batch` | 主进程是否还活着 |

monitor.py 输出样例：

```text
━━━ batch monitor  2026-04-30 22:42:13 ━━━
queue   total=10  done=4  error=1  pending=4  running=1
        [████▶▒    ]

active  groupnorm_1714485678_a8f3c2
        phase=EDIT  rounds=12/30  failures=1  plan_v=2
        baseline=18.421  best=14.012  speedup=1.31x
        heartbeat: 4s ago

        history (last 3 rounds):
          R10 keep    latency_us=1023  vectorize block_n
          R11 discard latency_us=1156  reorder loops
          R12 keep    latency_us=892   fuse epilogue

done speedup  median=1.42x  best=2.18x  worst=0.93x  (n=4)
              improved=3  on-par=0  regress=1

errored ops (1):
  - foo_kernel: phase=EDIT rc=0
```

### Step 6 — `summarize.py`：跑完离线汇总

```bash
python .autoresearch/scripts/batch/summarize.py $BATCH_DIR
```

跟 `monitor.py` 的区别：只读 `batch_progress.json`，不看实时状态，输出干净
可复制（贴 chat / ticket / 写日报用）。

---

## 常见操作

### 续跑 / 重试

总规律：

- `done` 不会重跑
- `error` 默认跳过；`--retry-errored` 捞回
- `pending` 自动续上
- `running`（上次跑挂留下的孤儿）下次启动时自动降级为 `error`

```bash
# 重试所有 errored op
python .autoresearch/scripts/batch/run.py $BATCH_DIR --worker-url 127.0.0.1:9111 --retry-errored

# 只跑某些 op
python .autoresearch/scripts/batch/run.py $BATCH_DIR --worker-url 127.0.0.1:9111 --only opA,opB

# 只跑前 N 个
python .autoresearch/scripts/batch/run.py $BATCH_DIR --worker-url 127.0.0.1:9111 --limit 5

# 跳过某个 op（修改 progress 把它标 skip）
# 或者用 --only 间接跳过
```

### 单 op debug

不用 batch，直接进 `claude` 交互模式：

```bash
cd /path/to/claude-autoresearch
claude
```

粘命令：

```bash
/autoresearch --ref $BATCH_DIR/refs/<op>_ref.py --kernel $BATCH_DIR/kernels/<op>_kernel.py \
  --op-name <op> --dsl triton_ascend --worker-url 127.0.0.1:9111 --max-rounds 30
```

⚠️ scaffold 打 `Task directory created: <path>` 后**立刻**让 Claude 跑
`export AKG_AGENTS_AR_TASK_DIR="<path>"` —— 这条 export 激活 hook 链，没跑下游全废。
batch 模式的 prompt 模板会反复强调这点；手动模式得自己注意。

### 中途介入

| 场景 | 怎么办 |
|------|--------|
| 暂停整个批量 | `tmux kill-session -t ar_batch`，当前 op 标 error |
| 跳过某个 op | 编辑 `batch_progress.json` 把 status 改 `skip` |
| 重试某个 errored op | `run.py --only <op> --retry-errored` |
| 清掉陈旧 ar_tasks | `rm -rf <repo>/ar_tasks/*`（**只在 run.py 没跑时**） |
| 换设备 | kill tmux → 改 worker daemon `--devices` → 重起 run.py |

### 子集预览（不进 manifest）

```bash
python .autoresearch/scripts/batch/discover.py $BATCH_DIR                  # 一行一个
python .autoresearch/scripts/batch/discover.py $BATCH_DIR --json           # JSON 数组
python .autoresearch/scripts/batch/discover.py $BATCH_DIR --filter '*norm' # 筛选预览
```

---

## Batch 目录布局

```text
<batch_dir>/                      ← 传给 run.py / monitor.py
├── manifest.yaml                 # prepare.py 写（ref_dir / kernel_dir / ops 列表）
├── batch_progress.json           # run.py 写：每个 op 的 status / task_dir / metrics
├── batch.log                     # run.py 写：claude --print 全部 stdout
├── verify_results.json           # prepare.py / verify.py 写
├── refs/<op>_ref.py              # ⚠️ 命名严格
└── kernels/<op>_kernel.py        # ⚠️ 命名严格

<repo>/ar_tasks/<op>_<ts>_<uuid>/ ← /autoresearch 自己维护的 round 级
├── kernel.py                     # 当前最佳 kernel
├── reference.py                  # scaffold 拷过来的 ref
├── task.yaml                     # arch / dsl / metric 配置
└── .ar_state/
    ├── .phase                    # 结束时 FINISH
    ├── progress.json             # eval_rounds / baseline_metric / best_metric
    ├── plan.md                   # agent 优化历史
    ├── history.jsonl             # 每轮 keep/discard 决策
    └── report.md                 # 最终报告（含 SVG）
```

**两层穿起来**：`batch_progress.json:cases.<op>.task_dir` 字段指向对应的
`ar_tasks/` 路径。

`manifest.yaml` 示例：

```yaml
mode: ref-kernel
dsl: triton_ascend
ref_dir: refs
kernel_dir: kernels
ops:
  - avgpool2d
  - batchnorm
  - groupnorm
```

---

## 收集优化后的 kernel

```bash
mkdir -p /tmp/optimized_kernels
python -c "
import json, shutil
from pathlib import Path
prog = json.load(open('$BATCH_DIR/batch_progress.json'))
for k, v in prog['cases'].items():
    if v.get('status') == 'done' and v.get('task_dir'):
        src = Path(v['task_dir']) / 'kernel.py'
        if src.exists():
            shutil.copy(src, f'/tmp/optimized_kernels/{k}.py')
            print('copied', k)
"
```

归档建议把整个 `<batch_dir>` + 每个 done op 对应的 `ar_tasks/` 子目录一起打包
（report.md / plan.md / history.jsonl 都在那）。

---

## `run.py` 全部参数

```text
位置参数：
  batch_dir                batch 目录路径，下面要有 manifest.yaml/json

硬件（XOR，默认 worker-url=127.0.0.1:9111 + 启动 health check）：
  --devices N              本地 NPU 设备 id（in-process eval）
  --worker-url host:port   worker daemon URL

透传给 /autoresearch：
  --max-rounds 30          每个 op 最多多少轮
  --eval-timeout 600       单次 eval 超时（秒）

batch 自己的兜底：
  --timeout-min 180        单 op 整体 wall-clock 上限（分钟），超时标 error 下一个

队列筛选：
  --only A,B,C             只跑指定 op
  --limit N                只跑前 N 个（0=不限）
  --retry-errored          也把 status=error 的算入队列

调度：
  --cooldown-sec 5         op 之间 sleep（0=关闭）

Claude CLI 透传：
  --claude-bin claude      claude 可执行文件路径
  --model ""               指定 model（空=默认）
  --extra-claude-arg ...   额外参数（可重复多次）
```

---

## 命令速查

```bash
# === 预备 ===
python .autoresearch/scripts/batch/prepare.py <batch_dir> --dsl triton_ascend
python .autoresearch/scripts/batch/prepare.py <batch_dir>                                    # 沿用已有 dsl
python .autoresearch/scripts/batch/prepare.py <batch_dir> --filter '*norm' --exclude 'foo*'
python .autoresearch/scripts/batch/prepare.py <batch_dir> --skip-verify                      # 只 discover

# === Worker daemon (akg 主体提供) ===
akg_cli worker --start --backend ascend --arch <your-ascend-arch> --devices 0 --port 9111
akg_cli worker --start --backend ascend --arch <your-ascend-arch> --devices 3 --port 9111
akg_cli worker --stop

# === 跑批量 ===
tmux new -d -s ar_batch \
  'python -u .autoresearch/scripts/batch/run.py <batch_dir> --worker-url 127.0.0.1:9111'
python .autoresearch/scripts/batch/run.py <batch_dir> --worker-url 127.0.0.1:9111 --only opA,opB
python .autoresearch/scripts/batch/run.py <batch_dir> --worker-url 127.0.0.1:9111 --retry-errored
python .autoresearch/scripts/batch/run.py <batch_dir> --worker-url 127.0.0.1:9111 --max-rounds 50

# === 监控 / 汇总 ===
python .autoresearch/scripts/batch/monitor.py <batch_dir>                # watch loop
python .autoresearch/scripts/batch/monitor.py <batch_dir> -n 10          # 改刷新间隔
python .autoresearch/scripts/batch/monitor.py <batch_dir> --dashboard    # 钻进 active task TUI
python .autoresearch/scripts/batch/summarize.py <batch_dir>              # 静态汇总
tail -f <batch_dir>/batch.log
tmux attach -t ar_batch
```

---

## 环境要求

| 角色 | 路径 / 要求 |
|------|------------|
| Batch 脚本 | `<repo>/.autoresearch/scripts/batch/{prepare,run,monitor,verify,summarize,discover,manifest}.py` |
| Batch 目录 | 用户自选 |
| 自动产物 | `batch_progress.json` / `batch.log` / `verify_results.json` |
| Autoresearch 任务输出 | `<repo>/ar_tasks/<op>_<ts>_<uuid>/` |
| Worker daemon log | `/tmp/ar_worker_<port>.log` |
| `claude` CLI | 必须在 `PATH`，或 `--claude-bin` 指定 |
| pyyaml | 可选；不装的话 manifest 用 JSON 格式 |
| torch / torch_npu | `/autoresearch` 实跑要；静态预检不要 |

---

## 故障排查

### `worker daemon at 127.0.0.1:9111 is unreachable`

`curl http://127.0.0.1:9111/api/v1/status` 看状态；没起就在 worker 机器上
`akg_cli worker --start --backend ascend --arch <your-ascend-arch> --devices 0 --port 9111`。
或者改用 `--devices N` 走 in-process eval（slower for batch）。

### phase 卡死、`--retry-errored` 也救不回来

模型偶尔漏跑 `export AKG_AGENTS_AR_TASK_DIR`，导致 hook 链没激活。`run.py` 的 prompt
反复强调这一步，仍偶发。直接手动接管：进 `claude` 交互模式 + `/autoresearch --resume <task_dir>`。

### `claude --print` 启动失败

`which claude` 看是否在 PATH；不在加 `--claude-bin /full/path`。

### 单 op wall-clock 超时

默认 `--timeout-min 180`（3 小时）。复杂 op 不够 → 加大到 `--timeout-min 300`。

### Pre-flight 报 `<file> not found` 但文件明明在

文件名是否严格 `<op>_ref.py` / `<op>_kernel.py`？`layernorm.py` ❌、
`layernorm_ref.py` ✅。

### 装了 pyyaml 但说 `pyyaml not installed`

多半装到别的 conda env 了。`python -c "import yaml; print(yaml.__file__)"` 确认。

### `--worker-url` 模式 baseline 报 `proxy connection refused`

`ALL_PROXY` 劫持了 127.0.0.1。`run.py` 启动 claude 时已经强制
`NO_PROXY=127.0.0.1,localhost`，但你自己起 worker daemon 那个终端可能没 set：

```bash
export NO_PROXY=127.0.0.1,localhost
# 重起 worker daemon
```

### Windows 跑 verify 报 `OMP: Error #15: libiomp5md.dll`

PyTorch + NumPy MKL 双初始化（Windows + PyTorch 通病）。verify.py 已经默认
`KMP_DUPLICATE_LIB_OK=TRUE` 解决；如果仍报，看是不是被环境变量显式覆盖成
`FALSE` 了。Linux/NPU 环境无此问题。
