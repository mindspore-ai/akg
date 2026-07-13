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

## 迁移指引

- 新功能用 `core_v2` 中的对应模块
- 需要修改上述旧模块时，优先考虑是否可以在 `core_v2` 中重新实现
- `async_pool` 和 `worker` 系列目前没有 core_v2 对应物，暂时保留在 core 中

## Worker 生命周期约束

- 本地 verify/profile 子进程必须独占进程组；超时或协程取消时按
  `SIGTERM -> grace -> SIGKILL` 回收整个进程组，不能只看组长进程的
  `returncode`，因为组长可能先退出而孙进程仍持有管道和设备。
- 在线程池执行的外部 profiler（msprof/nsys）必须接收协作取消信号；调用协程
  只有在线程完成子进程组清理并退出后才能继续释放设备租约。
- 远端设备租约由 `(device_id, lease_id)` 唯一标识。新客户端的每个工作请求
  必须携带精确 token；`task_id` 只用于可读身份和旧客户端兼容，不能作为并发
  续租的唯一依据。
- worker daemon 运行中的请求通过 keepalive 延长精确租约；请求结束后客户端
  负责 release，客户端失联则由 TTL reaper 回收。
- local/remote profile 的失败返回共用 `core.worker.interface.empty_profile_result`
  结构，缺测量统一使用 `None`，禁止在 HTTP/JSON 边界返回 `inf`/`nan`。
