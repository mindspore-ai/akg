# AKG Agent — 工作空间

> 本目录是 AKG Agents 的**使用态工作空间**（`akg_agents/workspace/`）。
> 在此目录下打开 OpenCode / Cursor / Claude Code 即可使用。
> 开发 akg_agents 代码本身请在上级目录 `akg_agents/` 下操作。

## 任务路由

所有 Agent 和 Skill 定义在 `.opencode/agents/` 和 `.opencode/skills/` 下。通过软链接（`.claude/ → .opencode/`、`.cursor/skills/ → ../.opencode/skills/`），OpenCode / Claude Code / Cursor 三个工具均可自动发现和加载，无需手动引用。

## 通用规则

以下规则对所有 agent / skill 生效。

- **环境准备、路径约定、变量定义、Shell 规则** → 由 `akg-env-setup` skill 在首次使用时自动引导完成，结果缓存到 `~/.akg/check_env.md`
- **算子优化 / 融合分析** → 切换到 `akg-op` agent3
- **安装配置** → 切换到 `akg-installer` agent
- **参数校验** → 所有传给 akg_agents API / 脚本 / akg_cli 的参数必须是以下有效值，构建命令前逐项校验：
| 参数 | 有效值 |
|------|--------|
| `framework` | `torch`, `mindspore` |
| `backend` | `cuda`, `ascend`, `cpu` |
| `dsl` | `triton_cuda`, `triton_ascend`, `cpp`, `cuda_c`, `tilelang_cuda`, `ascendc`, `pypto` |
| `arch` | cuda: `a100`, `v100`；ascend: `ascend910b1`~`ascend910b4`, `ascend310p3`；cpu: `x86_64`, `aarch64` |

各 agent/skill 内部已包含完整的流程说明、参数约束和禁止行为，不在此重复。