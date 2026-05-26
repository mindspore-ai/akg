# workspace_autoresearch

Claude Code 驱动的算子迭代优化工作区，集成在 `akg_agents/` 下。给一对
`(reference.py, kernel.py)`，Claude 自动跑 **plan → edit → eval → KEEP/DISCARD**
循环把 kernel 性能调优，连续失败自动 DIAGNOSE，预算耗尽自动收尾出报告。
整套阶段机由 hook 强约束，Claude 不能跳步、不能改 plan.md、不能手写 phase。

支持 DSL（端到端验证状态以 `triton_ascend` 为参考；其它 DSL 走 akg 主体相同
代码路径，但 NPU/CUDA 真实硬件覆盖度依本仓 CI 为准）：

| DSL | 后端 | skills 树 (`op/resources/skills/<dsl>/`) |
|---|---|---|
| `triton_ascend`  | Ascend | ✓ |
| `triton_cuda`    | CUDA   | ✓ |
| `tilelang_cuda`  | CUDA   | ✓ |
| `tilelang_npuir` | Ascend | — |
| `ascendc`        | Ascend | — |
| `cuda_c`         | CUDA   | ✓ |
| `cpp`            | CPU    | ✓ |
| `pypto`          | Ascend | ✓ |
| `swft`           | Ascend | — |
| `torch`          | CPU    | — |

## 与 akg_agents 的关系

本目录是 Claude Code 表面层。底层 verifier / DSL 适配器 / 补丁等核心实现
直接来自 `akg_agents.op.verifier` 与 `akg_agents.op.utils`，通过 Python
import 复用。skills 知识库共用 `akg_agents/python/akg_agents/op/resources/skills/`
（通过 `.claude/settings.json` 的 `AKG_AGENTS_AR_SKILLS_ROOT` 环境变量定位）。

本目录独有的部分：phase machine + workflow + hooks + slash command —— Claude
Code 特定的交互式优化编排，akg 主线的 LangGraph workflow 不复用这一层。

## 依赖

- Python ≥ 3.10
- `git` 可执行（部分精简镜像不自带；workspace 依赖 git 做 per-round commit/rollback）
- `pip install pyyaml fastapi uvicorn`
- akg_agents 可导入：在 `akg_agents/` 根 `pip install -e .`，或确保
  `akg_agents/python/` 在 `PYTHONPATH` 上
- Claude Code CLI（或 VS Code 扩展）
- 按 DSL 追加：`torch_npu` + `triton` + CANN（Ascend）；`triton` + CUDA runtime（CUDA）；
  `msprof` 在 PATH（ascendc）；`nsys` 在 PATH（cuda_c）

## Quick Start

```bash
cd akg_agents/workspace_autoresearch
# 把 (<op>_ref.py, <op>_kernel.py) 放到 workspace/
claude
```

在 Claude 里粘 slash 命令：

```bash
/autoresearch --ref workspace/sinkhorn_ref.py --kernel workspace/sinkhorn_kernel.py \
  --op-name sinkhorn --dsl triton_ascend --devices 5 --max-rounds 200
```

scaffold + 首轮 baseline 原子完成 → 进 PLAN → 自动迭代到 FINISH。

监控：

```bash
python .autoresearch/scripts/dashboard.py
```

## Worker

远端 NPU / CUDA 通过 SSH tunnel 接入，eval 提交到远端跑。worker 直接用 akg 的
`akg_cli`（worker 端只要装好 akg_agents 即可）：

```bash
# worker 机器
akg_cli worker --start --backend ascend --arch <your-ascend-arch> --devices 0 --port 9111

# 本地建隧道
ssh -f -N -L 127.0.0.1:9111:127.0.0.1:9111 npu_host
curl http://127.0.0.1:9111/api/v1/status
```

task.yaml 里写 `worker.urls: [127.0.0.1:9111]`，workspace 的 eval 自动通过
`akg_agents.core.worker.RemoteWorker` 发到远端。

## 输出

每个 task 落在 `ar_tasks/<op>_<ts>_<uuid>/`：

```text
ar_tasks/<op>_<ts>_<uuid>/
├── kernel.py          ← 性能优化后的 kernel
├── reference.py       ← scaffold 拷过来的 ref
├── task.yaml          ← dsl / arch / metric / editable_files
└── .ar_state/
    ├── .phase         ← 当前 phase（结束时是 FINISH）
    ├── progress.json
    ├── plan.md        ← agent 优化历史（权威态）
    ├── history.jsonl
    └── report.md      ← 含 SVG 收敛曲线
```

## 内部入口

| 想了解 | 看哪里 |
|--------|--------|
| Phase 流转规则 / Bash gate | [phase_machine/phase_policy.py](.autoresearch/scripts/phase_machine/phase_policy.py) |
| Hook 接线 | [.claude/settings.json](.claude/settings.json) + [hooks/](.autoresearch/scripts/hooks/) |
| Plan / history / progress 写入 | [phase_machine/state_store.py](.autoresearch/scripts/phase_machine/state_store.py) |
| 评测桥（KernelVerifier + WorkerManager 包装为 sync dict）| [utils/akg_eval.py](.autoresearch/scripts/utils/akg_eval.py) |
| DSL adapter / Profiler / Verify / Worker | `akg_agents.op.verifier.*` / `akg_agents.core.worker.*`（akg 主体）|
| CodeChecker（async 包装为 sync）| [utils/code_checker.py](.autoresearch/scripts/utils/code_checker.py) → akg `op.utils.code_checker` |
| Git 操作（GitRepo 包装）| [utils/git_utils.py](.autoresearch/scripts/utils/git_utils.py) → akg `op.autoresearch.framework.git_repo.GitRepo` |
| 不变量（plan 权威态 / pid 单调 / DIAGNOSE 契约 等） | [AGENTS.md](AGENTS.md) |
| 子代理 prompt（DIAGNOSE 用） | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |
| 批量跑 | [BATCH.md](BATCH.md) |
