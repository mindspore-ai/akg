# AKG Agents

你是专用于 AKG Agents 的开发 Code Agent。请基于本文档（AGENTS.md）以及各目录下分级的 SPEC.md 规格文档来开发 AKG Agents 框架。

AKG Agents 是基于 LLM 的多 Agent 协作框架，面向 AI Infra 和高性能计算。当前核心场景为多后端、多 DSL 内核代码生成，后续将扩展图优化等更上层编译能力。

> **使用态**（以 KernelAgent 做算子生成/优化）请到 `workspace/` 目录打开 code agent。
> 本文件面向**开发态**——开发 akg_agents 代码本身。

---

## 开发前必读

**开始开发前，建议阅读目标目录及其上级目录的 SPEC.md**，以快速了解每个目录的关键结构与代码开发要求。SPEC.md 描述了该目录"做什么、不做什么、怎么做"。

**涉及关键性修改时，需同步更新对应层级的 SPEC.md**：
- 新增/删除模块 → 更新该目录的 SPEC.md 中的目录结构和子目录索引
- 新增/变更对外接口或基类 → 更新 SPEC.md 中的开发约定
- 涉及跨目录的架构调整 → 更新本文档（AGENTS.md）的目录索引和全局规范

---

## 快速开始

```bash
pip install -r requirements.txt && pip install -e ./ --no-build-isolation
source env.sh          # 与 -e 安装二选一
akg_cli --help         # 验证安装

./run_test.sh -t ut    # 单元测试（不需要 LLM/GPU）
```

---

## 目录索引

### 核心源码（python/akg_agents/）

| 目录 | 职责 | 规范 |
|------|------|------|
| `core_v2/` | v2 核心框架（Agent、Workflow、Skill、Tool、LLM、Config） | [SPEC.md](python/akg_agents/core_v2/SPEC.md) |
| `op/` | 算子/内核生成场景层 | [SPEC.md](python/akg_agents/op/SPEC.md) |
| `cli/` | akg_cli 命令行入口（Typer） | [SPEC.md](python/akg_agents/cli/SPEC.md) |
| `core/` | 旧版核心（迁移中，不要新增代码） | [SPEC.md](python/akg_agents/core/SPEC.md) |
| `utils/` | 跨模块共享工具 | [SPEC.md](python/akg_agents/utils/SPEC.md) |

其余子包：`database/`（数据库/向量库）、`server/` + `client/` + `worker/`（远程服务）、`config/`（全局配置入口）、`resources/`（prompts/skills/docs/templates）、`tool/`（code agent 工具定义）。

### 仓库根目录

| 目录 | 职责 | 规范 |
|------|------|------|
| `tests/` | 测试 | [SPEC.md](tests/SPEC.md) |
| `docs/` | 设计文档 | [SPEC.md](docs/SPEC.md) |
| `examples/` | 使用示例 | [SPEC.md](examples/SPEC.md) |
| `tools/` | 辅助批跑/检查工具 | [SPEC.md](tools/SPEC.md) |
| `benchmark/` | 评测集 | [SPEC.md](benchmark/SPEC.md) |
| `workspace/` | 使用态工作空间（KernelAgent） | [README.md](workspace/README.md) |

其余目录：`akg-cli/`（npm CLI 客户端包）、`scripts/`（构建/发布辅助脚本）、`thirdparty/`（KernelBench 等第三方子模块）。

---

## 全局代码规范

### License 头

所有 `.py` 文件必须包含 Apache 2.0 License 头：

```python
# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

### 包结构

- Python >= 3.10
- 源码在 `python/` 目录下（`package_dir={"": "python"}`）
- 新目录必须放 `__init__.py`
- 资源文件（`.j2`、`.md`、`.yaml`、`.json`）通过 `setup.py` 的 `package_data` 打包

### 环境

- `env.sh` 只做一件事：`export PYTHONPATH=$(pwd)/python:$PYTHONPATH`
- 每次新 shell 都需要 `source env.sh`
- 版本号在 `version.txt`

---

## 参数有效值

所有传给 akg_agents API / akg_cli 的参数必须使用以下规范值：

| 参数 | 有效值 |
|------|--------|
| `framework` | `torch`, `mindspore` |
| `backend` | `cuda`, `ascend`, `cpu` |
| `dsl` | `triton_cuda`, `triton_ascend`, `cpp`, `cuda_c`, `tilelang_cuda`, `ascendc`, `pypto` |
| `arch` | cuda: `a100`, `v100`；ascend: `ascend910b1`~`ascend910b4`, `ascend310p3`；cpu: `x86_64`, `aarch64` |

---

## Shell 命令规则

每条 shell 命令运行在独立 session，环境激活不跨命令持久化。

**conda 环境**（`$ENV_TYPE=conda`）：
```bash
conda run -n $CONDA_ENV --no-capture-output bash -c \
  "cd $AKG_AGENTS_DIR && source env.sh && <CMD>"
```

**venv 环境**（`$ENV_TYPE=venv`）：
```bash
bash -c "source $VENV_PATH/bin/activate && cd $AKG_AGENTS_DIR && source env.sh && <CMD>"
```

> 如果已有 `$HOME_DIR/.akg/check_env.md`，直接使用其中的「命令模板」字段，将 `<CMD>` 替换为实际命令即可。

### 禁止行为

| 行为 | 级别 |
|------|------|
| 裸执行 `pip`/`python`（未激活环境） | ⛔ 致命 |
| 依赖 `conda activate` 或 `source activate` 跨命令持久化 | ⛔ 致命 |
| 不 `source env.sh` 就运行 akg_agents 相关脚本 | ❌ 错误 |
| 用 `echo 'y' \|` 管道代替 `--yes` 标志 | ⛔ 致命 |

---

## 环境检查 Skill

| Skill | 用途 | 加载场景 |
|-------|------|---------|
| `akg-env-setup` | 环境检查 + 采集 + 缓存；FULL_SETUP 模式额外含当次任务的参数确认和运行时依赖安装 | 安装请求（基础模式）；op-optimizer Phase 1（FULL_SETUP 模式） |
| `akg-pr` | 基于当前分支与目标分支的 diff 生成 PR 描述文件（.md + .json），写入 `.tmp/pr/` | 用户输入 `/akg_pr` |
| `akg-issue` | 辅助构建 Issue 描述文件（Bug Report / RFC / Task），写入 `.tmp/issue/` | 用户输入 `/akg_issue` |

---

## PR / Issue 生成规则

### `/akg_pr`

基于当前分支与目标分支（默认 `master`）的 diff，自动生成符合 AKG 规范的 PR 描述。

**产物路径**：`$AKG_AGENTS_DIR/.tmp/pr/`
- `pr_<branch>_<YYYYMMDD_HHmmss>.md` — PR 描述，遵循 `.github/PULL_REQUEST_TEMPLATE.md` 格式
- `pr_<branch>_<YYYYMMDD_HHmmss>.json` — 元数据（title, kind, labels, target_branch, validation 等）

**校验规则**（生成后自动执行）：
- `/kind` 必须为 `bug`/`task`/`feature` 之一
- PR 描述至少 20 字
- bug fix PR 必须关联 issue（`Fixes #`）
- 可通过 `scripts/validate_pr_issue.py <json_file>` 执行校验

### `/akg_issue`

辅助构建 Issue，支持 Bug Report / RFC / Task 三种类型。

**产物路径**：`$AKG_AGENTS_DIR/.tmp/issue/`
- `issue_<slug>_<YYYYMMDD_HHmmss>.md` — Issue 描述，遵循 `.github/ISSUE_TEMPLATE/` 对应模板格式
- `issue_<slug>_<YYYYMMDD_HHmmss>.json` — 元数据（title, issue_type, labels, validation 等）

**校验规则**（生成后自动执行）：
- title 至少 10 字符
- 必须包含 `kind/*` 标签
- Bug Report 必须包含环境信息、当前/期望行为、复现步骤
- 可通过 `scripts/validate_pr_issue.py <json_file>` 执行校验
