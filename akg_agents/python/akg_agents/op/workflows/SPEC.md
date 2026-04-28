# op/workflows/ — 算子工作流

## 职责

定义算子/内核生成的 LangGraph 工作流。每个 workflow 由节点（Agent 调用）和边（路由逻辑）组成。

## 继承体系

```
BaseWorkflow (core_v2)
└── OpBaseWorkflow                      # base_workflow.py — 算子工作流基类
    ├── DefaultWorkflow                 # default_workflow.py
    ├── DefaultWorkflowV2               # default_workflow_v2.py
    ├── CoderOnlyWorkflow               # coder_only_workflow.py
    ├── KernelGenOnlyWorkflow           # kernelgen_only_workflow.py
    ├── VerifierOnlyWorkflow            # verifier_only_workflow.py
    ├── ConnectAllWorkflow              # connect_all_workflow.py
    ├── EvolveWorkflow                  # evolve_workflow.py
    └── AdaptiveSearchWorkflow          # adaptive_search_workflow.py

OpTaskBuilderWorkflow                   # op_task_builder_workflow.py（独立封装，不继承 BaseWorkflow）
```

## WORKFLOW_REGISTRY

`LangGraphTask`（`langgraph_op/task.py`）中注册了所有可用 workflow，短名和长名均可使用：

| 短名 | 长名 | 类 |
|------|------|-----|
| `default` | `default_workflow` | `DefaultWorkflow` |
| `default_v2` | `default_workflow_v2` | `DefaultWorkflowV2` |
| `coder_only` | `coder_only_workflow` | `CoderOnlyWorkflow` |
| `kernelgen_only` | `kernelgen_only_workflow` | `KernelGenOnlyWorkflow` |
| `verifier_only` | `verifier_only_workflow` | `VerifierOnlyWorkflow` |
| `connect_all` | `conductor_connect_all_workflow` | `ConnectAllWorkflow` |
| `evolve` | `evolve_workflow` | `EvolveWorkflow` |
| `adaptive_search` | `adaptive_search_workflow` | `AdaptiveSearchWorkflow` |

## 开发约定

### 新增 workflow 的标准流程

1. 在本目录创建 `<workflow_name>.py`
2. 继承 `OpBaseWorkflow`
3. 实现 `build_graph()` 定义节点和边
4. 在 `langgraph_op/task.py` 的 `WORKFLOW_REGISTRY` 中注册
5. 如需配套配置，在 `../config/` 下添加 yaml

### OpBaseWorkflow 核心 API

- `build_graph()` → 定义工作流图（节点 + 边）
- `build_langgraph_task_config()` → 构建 LangGraphTask 配置

## 不做什么

- **不要**在 workflow 中实现具体的 Agent 逻辑——Agent 归 `../agents/`
- **不要**在 workflow 中直接调用 LLM——通过 Agent 间接调用
- **不要**修改 `OpBaseWorkflow` 的核心接口——通过继承扩展
