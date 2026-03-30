# tools/ — 辅助工具

## 职责

独立于主包的辅助脚本，用于批量执行、任务构建、环境检查等运维操作。

## 目录结构

| 子目录/文件 | 说明 |
|------------|------|
| `v2/use_llm_check/` | LLM 连通性检查工具（`test_run_llm.py`） |

> 批量执行、任务构建、随机验证等脚本已统一迁移至 `python/akg_agents/op/tools/`。

## 开发约定

- 脚本可以 `import akg_agents`（需要先 `source env.sh`）
- 脚本应有 `--help` 说明或文件头注释
- 批跑脚本的输出目录统一使用 `~/akg_agents_logs/`

## 不做什么

- **不要**把主包的核心逻辑放在这里——归 `python/akg_agents/`
- **不要**把测试放在这里——归 `tests/`
