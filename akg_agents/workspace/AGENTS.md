# AKG Agent — 使用指南

> 本目录是 AKG Agents 的**使用态工作空间 akg_agents/workspace/**。
> 在此目录下打开 OpenCode / Cursor / Claude Code 即可使用 AKG Agent 功能。
>
> 如需开发 akg_agents 代码本身，请在上级目录 `akg_agents/` 下打开你的 code agent。

---

## 工作方式

所有 Agent 和 Skill 定义在 `.opencode/agents/` 和 `.opencode/skills/` 下，由 code agent 自动加载，无需手动引用。

- **环境准备、路径约定、变量定义、参数校验、Shell 规则** → 由 `akg-env-setup` skill 在首次使用时自动引导完成，结果缓存到 `~/.akg/check_env.md`
- **算子优化 / 融合分析** → 切换到 `op-optimizer` agent
- **安装配置** → 切换到 `akg-installer` agent

各 agent/skill 内部已包含完整的流程说明、参数约束和禁止行为，不在此重复。
