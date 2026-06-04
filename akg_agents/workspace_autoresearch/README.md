# workspace_autoresearch

Claude Code 驱动的算子迭代优化工作空间，`akg_agents/` 的使用态目录。给一对
`(<op>_ref.py, <op>_kernel.py)`，Claude 自动跑 `plan → edit → eval →
KEEP/DISCARD` 循环优化 kernel；连续失败自动 DIAGNOSE，达 `max_rounds` 出报告。
phase machine 由 hook 强约束，不能跳步、不能手写 plan.md / phase。

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
| [AUTORESEARCH.md](AUTORESEARCH.md) | 框架完全手册：A 单机 / B 双机 / C 批量 / D 续跑，加 §1–§11 参考章 |
| [AGENTS.md](AGENTS.md) | Claude Code agent 运行时不变量（plan 权威态 / pid 单调 / DIAGNOSE 契约） |
| [.claude/commands/autoresearch.md](.claude/commands/autoresearch.md) | `/autoresearch` slash 命令规范 |

## 最小启动

```bash
cd akg_agents/workspace_autoresearch
# 把 (<op>_ref.py, <op>_kernel.py) 放到 workspace/
claude
```

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <NPU-id>
```

监控：`python scripts/dashboard.py <task_dir> --watch`。
