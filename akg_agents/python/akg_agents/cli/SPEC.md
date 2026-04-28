# cli/ — akg_cli 命令行入口

## 职责

基于 Typer 框架的 `akg_cli` 命令注册、参数解析、运行时环境和流式 UI。入口为 `cli.py` 中的 `app`。

## 目录结构

| 子目录 | 职责 |
|--------|------|
| `commands/` | 子命令定义（`op/`、`trace`、`slash_commands`、`common`、`misc`、`resume` 等） |
| `service/` | CLIAppServices、WorkerService、参数规范化 |
| `runtime/` | ReAct 执行器、通用工具注册、补丁 |
| `stream/` | 流式渲染、JSON 处理、日志、安全过滤 |
| `ui/` | 补全、进度条、面板、插件注册 |
| `utils/` | 路径解析、设备检测、文案 |

## 开发约定

### 新增命令的标准流程

1. 在 `commands/` 下创建命令函数（或子目录）
2. 在 `cli.py` 中通过 `app.command()` 或 `register_*_command` 注册
3. 参数规范化在 `service/normalization.py` 中处理
4. 业务逻辑通过调用 `op/` 或 `core_v2/` 的 API 完成

## 不做什么

- **不要**在此实现业务逻辑（算子生成、验证等）——归 `op/`
- **不要**在此定义 Agent / Workflow 基类——归 `core_v2/`

## 参考

- `docs/v2/AKG_CLI.md` — CLI 命令参考
