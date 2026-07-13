# workspace_autoresearch

编码 agent(Claude Code / opencode)驱动的算子迭代优化工作空间，`akg_agents/`
的使用态目录。给一对 `(<op>_ref.py, <op>_kernel.py)`，agent 自动跑
`plan → edit → eval → KEEP/DISCARD` 循环优化 kernel；连续失败自动 DIAGNOSE，
达 `max_rounds` 出报告。phase machine 由 hook 强约束，不能跳步、不能手写
plan.md / phase。两种 agent 共用同一 `scripts/decide.py` 决策脑。

## 与 akg_agents 的关系

phase machine + hooks + slash command 在本目录 `scripts/`；verifier 与 worker
直接 import `akg_agents.op.verifier.KernelVerifier` 和
`akg_agents.core.worker.manager`，桥在
[scripts/utils/akg_eval.py](scripts/utils/akg_eval.py)。skills 树共用
`akg_agents/python/akg_agents/op/resources/skills/`（通过
`.claude/settings.json` 的 `AKG_AGENTS_AR_SKILLS_ROOT` 定位）。

## 入口文档

| 文档 | 用途 |
|---|---|
| [AUTORESEARCH.md](AUTORESEARCH.md) | 精简使用指南：本地运行、远端 worker、批量、恢复和 trace。 |
| [AGENTS.md](AGENTS.md) | agent 运行时不变量（Claude / opencode 共享：plan 权威态 / pid 单调 / DIAGNOSE 契约） |
| [.claude/commands/autoresearch.md](.claude/commands/autoresearch.md) | `/autoresearch` slash 命令规范 |

## 最小启动

把 `(<op>_ref.py, <op>_kernel.py)` 放到 `workspace/`，在 `claude` 或 opencode 里输入同一条命令:

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <dev-id>
```

各 agent 的启动和 headless 跑法见 [AUTORESEARCH.md §1](AUTORESEARCH.md)。

后端默认 Ascend NPU；改 [config.yaml](config.yaml) `defaults.backend` + `dsl` 可切到 cuda / cpu，`<dev-id>` 由对应平台的 `npu-smi` / `nvidia-smi` 决定。

监控：`python scripts/dashboard.py <task_dir> --watch`。
