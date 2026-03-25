# utils/ — 共享工具

## 职责

跨模块共享的工具函数和辅助资源，不含业务逻辑。

## 目录结构

| 子目录/文件 | 职责 |
|------------|------|
| `evolve/` | 进化搜索相关工具（runner_manager、processors、result_collector） |
| `langgraph/` | LangGraph 辅助（state、routers、visualizer、node_tracker） |
| `compile_tools/` | Ascend 编译辅助（CMake 模板、头文件、编译脚本） |
| `common_utils.py` | 通用工具函数 |
| 其他 `.py` | case_generator、collector、environment_check 等 |

## 关键约定

- **`get_prompt_path()`**（`common_utils.py`）固定指向 `op/resources/prompts/`——这是所有 Agent prompt 模板的根目录，不是本目录的 `resources/`

## 不做什么

- **不要**在此写业务逻辑
- **不要**在此写仅被单一模块使用的工具——放在对应模块的本地 utils 中
