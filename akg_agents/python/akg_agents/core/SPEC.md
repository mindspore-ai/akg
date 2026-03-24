# core/ — 旧版核心（迁移中）

## 状态：逐步废弃

新代码请使用 `core_v2/`。本目录中的模块正在逐步迁移，**不要在此新增代码**。

## 仍在被引用的模块

以下模块仍被其他代码广泛引用，短期内不会删除：

| 模块 | 被谁引用 | 说明 |
|------|---------|------|
| `core.async_pool.device_pool` | op/evolve、op/verifier、op/langgraph_op | 设备池管理 |
| `core.async_pool.task_pool` | op/evolve、op/workflows、op/tools | 任务池管理 |
| `core.worker.manager` | op/ 各工作流和 Agent、cli、server | Worker 注册与管理 |
| `core.worker.local_worker` | op/evolve、op/verifier、op/langgraph_op | 本地 Worker |
| `core.worker.remote_worker` | op/verifier、server | 远程 Worker |
| `core.worker.interface` | op/verifier | Worker 接口定义 |
| `core.agent.coder` | op/langgraph_op/task | 旧版 Coder Agent |
| `core.agent.designer` | op/langgraph_op/task | 旧版 Designer Agent |
| `core.agent.selector` | op/utils、op/skill | 旧版 Selector |
| `core.checker` | op/workflows | CodeChecker |
| `core.tools.basic_tools` | cli/runtime | 旧版工具 |
| `core.utils` | op/config、op/langgraph_op、op/verifier、cli | 工具函数 |
| `core.sketch` | op/adaptive_search、op/utils/evolve | Sketch 生成 |

## 迁移指引

- 新功能用 `core_v2` 中的对应模块
- 需要修改上述旧模块时，优先考虑是否可以在 `core_v2` 中重新实现
- `async_pool` 和 `worker` 系列目前没有 core_v2 对应物，暂时保留在 core 中
