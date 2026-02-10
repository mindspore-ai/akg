# 工作流系统设计文档

## 概述
AKG Agents 使用基于 **LangGraph** 的工作流系统定义和执行任务流程，能提供更好的灵活性、类型安全性和可视化能力。

## 架构总览

```
core_v2/langgraph_base/         # 通用基类（领域无关）
├── base_task.py                # BaseLangGraphTask - 任务基类
├── base_workflow.py            # BaseWorkflow - 工作流基类
└── visualizer.py               # 工作流可视化

op/langgraph_op/                # 算子领域实现
├── task.py                     # LangGraphTask - 算子任务执行器
├── state.py                    # KernelGenState - 算子状态定义
├── nodes.py                    # NodeFactory - 节点工厂
└── routers.py                  # RouterFactory - 路由工厂

op/workflows/                   # 算子工作流定义
├── base_workflow.py            # OpBaseWorkflow - 算子工作流基类
├── default_workflow.py         # 标准：Designer → Coder ↔ Verifier
├── coder_only_workflow.py      # 仅编码：Coder ↔ Verifier
├── kernelgen_only_workflow.py  # Skill 系统：KernelGen ↔ Verifier
├── verifier_only_workflow.py   # 仅验证：Verifier → END
└── connect_all_workflow.py     # 全连接：All ↔ All
```

## 核心概念

### BaseLangGraphTask（通用任务基类）
位于 `core_v2/langgraph_base/base_task.py`，提供任务执行的通用框架，不包含任何领域专用逻辑。

**核心方法：**
| 方法 | 说明 |
|------|------|
| `_init_workflow()` | 抽象方法，子类实现工作流初始化 |
| `_prepare_initial_state()` | 抽象方法，子类实现初始状态准备 |
| `run()` | 执行任务，返回 `(success, final_state)` |
| `visualize()` | 生成 Mermaid/PNG 流程图 |

### BaseWorkflow（通用工作流基类）
位于 `core_v2/langgraph_base/base_workflow.py`，提供领域无关的工作流抽象。

**核心方法：**
| 方法 | 说明 |
|------|------|
| `build_graph()` | 抽象方法，子类实现具体的图结构定义 |
| `compile()` | 编译工作流图，返回可执行的 LangGraph 应用 |
| `visualize()` | 生成 Mermaid 格式的流程图 |

### LangGraphTask（算子任务执行器）
位于 `op/langgraph_op/task.py`，继承 `BaseLangGraphTask`，添加算子生成场景的专用逻辑：
- 算子 Agent 初始化（Designer, Coder, KernelGen, Verifier）
- 设备池和 Worker 管理
- 算子专用初始状态准备
- 工作流注册表（`WORKFLOW_REGISTRY`）

### KernelGenState（状态定义）
使用 `TypedDict` 定义的类型安全状态，包含：
- 通用字段：`task_id`, `iteration`, `max_iterations`, `success` 等
- 算子专用字段：`op_name`, `task_desc`, `dsl`, `framework`, `backend`, `arch` 等
- 结果字段：`verifier_result`, `verifier_error`, `coder_code`, `designer_sketch` 等

## 可用工作流

### 工作流注册表
```python
WORKFLOW_REGISTRY = {
    "default":           DefaultWorkflow,       # 标准流程
    "coder_only":        CoderOnlyWorkflow,     # 仅编码
    "kernelgen_only":    KernelGenOnlyWorkflow, # Skill 系统
    "verifier_only":     VerifierOnlyWorkflow,  # 仅验证
    "connect_all":       ConnectAllWorkflow,    # 全连接
}
```

### 1. 标准工作流 (`default`)
```
Designer → Coder → Verifier → [成功] → END
                       ↓ [失败]
                  Conductor → Coder（重复）
```
完整的设计→编码→验证流程。

### 2. 仅编码工作流 (`coder_only`)
```
Coder → Verifier → [成功] → END
            ↓ [失败]
       Conductor → Coder（重复）
```
跳过设计阶段，直接生成代码。

### 3. KernelGen 工作流 (`kernelgen_only`)
```
KernelGen → Verifier → [成功] → END
                ↓ [失败]
           Conductor → KernelGen（重复）
```
基于 Skill 系统的代码生成，支持动态知识注入。

### 4. 仅验证工作流 (`verifier_only`)
```
Verifier → END
```
最简化流程，仅验证现有代码。

### 5. 全连接工作流 (`connect_all`)
支持所有 Agent 间灵活跳转，最大灵活性。

## 工作流中的 Conductor 节点

Conductor 节点（非独立 Agent）在验证失败时执行 LLM 驱动的错误分析：
1. 加载 `conductor/analyze.j2` 模板
2. 准备包含错误信息、历史尝试等上下文的提示
3. 调用 LLM 进行分析
4. 解析决策（继续/终止）和建议
5. 返回更新的状态给下一个代码生成节点

## 快速开始

### 基本用法
```python
from akg_agents.op.langgraph_op.task import LangGraphTask

# 创建任务
task = LangGraphTask(
    op_name="akg_relu",
    task_desc="实现 ReLU 激活函数的框架代码...",
    task_id="0",
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    workflow="default"  # 选择工作流
)

# 运行任务
op_name, success, final_state = await task.run()
```

### 可视化
```python
# 打印 Mermaid 图
print(task.visualize())

# 保存为 PNG
task.visualize(output_path="workflow.png")
```

## 关键参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| op_name | str | 是 | 算子名称 |
| task_desc | str | 是 | 任务描述（框架代码）|
| task_id | str | 是 | 唯一任务标识符 |
| dsl | str | 是 | 目标 DSL：`triton_cuda`、`triton_ascend` 等 |
| framework | str | 是 | 前端框架：`torch`、`mindspore`、`numpy` |
| backend | str | 是 | 后端：`cuda`、`ascend` |
| arch | str | 是 | 硬件架构：`a100`、`ascend910b4` |
| config | dict | 是 | 配置字典 |
| workflow | str | 否 | 工作流名称（默认：`default`）|
| inspirations | list | 否 | 来自进化的代码灵感 |
| meta_prompts | str | 否 | LLM 的元提示 |
| user_requirements | str | 否 | 用户额外需求 |

## 自定义工作流

### 创建自定义工作流
```python
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from langgraph.graph import StateGraph, END

class MyWorkflow(OpBaseWorkflow):
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(KernelGenState)

        # 创建节点
        coder_node = NodeFactory.create_coder_node(
            self.agents['coder'], self.trace
        )
        verifier_node = NodeFactory.create_verifier_node(
            self.agents['verifier'], self.device_pool,
            self.trace, self.config
        )

        # 添加节点
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)

        # 定义流程
        workflow.set_entry_point("coder")
        workflow.add_edge("coder", "verifier")
        workflow.add_edge("verifier", END)

        return workflow
```

### 注册自定义工作流
在 `op/langgraph_op/task.py` 的 `WORKFLOW_REGISTRY` 中添加：
```python
WORKFLOW_REGISTRY = {
    ...
    "my_workflow": MyWorkflow,
}
```

或使用装饰器注册（推荐，支持 `KernelAgent` 动态发现）：
```python
from akg_agents.core_v2.workflows.registry import register_workflow

@register_workflow(scopes=["op"])
class MyWorkflow(OpBaseWorkflow):
    TOOL_NAME = "use_my_workflow"
    DESCRIPTION = "我的自定义工作流描述"
    PARAMETERS_SCHEMA = { ... }
    ...
```

## 工作流作为工具（Workflow as Tool）

工作流可以被 `KernelAgent`（ReAct Agent）作为工具调用。每个工作流类可以定义以下元数据：
- `TOOL_NAME`：工具名称（如 `use_kernelgen_only_workflow`）
- `DESCRIPTION`：工具描述（LLM 用于决策）
- `PARAMETERS_SCHEMA`：参数 JSON Schema

`KernelAgent` 通过 `WorkflowRegistry` 动态发现并加载注册了 `@register_workflow(scopes=["op"])` 的工作流，并将其暴露为可调用的工具。