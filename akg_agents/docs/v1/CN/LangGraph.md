# LangGraph 任务设计文档

## 概述
LangGraph Task 是新的智能任务调度器，用于替代原有的 `Conductor + workflow.yaml` 系统。它使用 LangGraph 框架在 Python 代码中定义工作流，提供更好的灵活性、可视化能力和类型安全性，同时保持完全向后兼容。

## 核心特性
- **Python 定义工作流**：使用代码而非 YAML 配置定义执行流程
- **图结构可视化**：生成工作流的 Mermaid 图和 PNG 图像
- **类型安全状态**：使用 `TypedDict` 提供更好的类型提示和 IDE 支持
- **向后兼容**：API 与原 `Task` 类完全一致，可无缝迁移

## 快速开始

### 基本用法
```python
from ai_kernel_generator.core.langgraph_task import LangGraphTask
from ai_kernel_generator.core.worker.manager import register_local_worker

# 注册 Worker
await register_local_worker([0], backend='cuda', arch='a100')

# 创建任务（API 与原 Task 完全相同）
task = LangGraphTask(
    op_name="aikg_relu",
    task_desc="实现 ReLU 激活函数",
    task_id="0",
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    device_pool=None,  # 使用 WorkerManager 替代
    workflow="default_workflow"
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

## 架构对比

| 组件 | 原架构（Task） | 新架构（LangGraphTask） |
|------|---------------|------------------------|
| **调度器** | `Conductor` 类 + `workflow.yaml` | LangGraph `StateGraph` + Python 工作流 |
| **工作流定义** | YAML 文件（`workflow_config_path`） | Python 类（`workflows/` 目录） |
| **Agent 配置** | 混合在 `workflow.yaml` 中 | 独立的 `parser_config.yaml` |
| **状态管理** | `task_info` 字典 | `KernelGenState` TypedDict |
| **决策逻辑** | Conductor LLM + 代码逻辑 | Conductor 节点 + Router 函数 |
| **可视化** | 无 | Mermaid / PNG 图 |

## 可用工作流

| 工作流名称 | 流程 | 描述 |
|-----------|------|------|
| `default_workflow` | Designer → Coder ↔ Verifier ↔ Conductor | 完整的设计→编码→验证流程 |
| `coder_only_workflow` | Coder ↔ Verifier ↔ Conductor | 跳过设计，直接生成代码 |
| `verifier_only_workflow` | Verifier → Finish | 仅验证现有代码 |
| `connect_all_workflow` | All ↔ All | 全连接 agents |

## 关键参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| op_name | str | 是 | 算子名称 |
| task_desc | str | 是 | 任务描述（框架代码） |
| task_id | str | 是 | 唯一任务标识符 |
| dsl | str | 是 | 目标 DSL："triton_cuda"、"triton_ascend"、"swft" |
| framework | str | 是 | 前端框架："torch"、"mindspore"、"numpy" |
| backend | str | 是 | 后端："cuda"、"ascend" |
| arch | str | 是 | 硬件架构："a100"、"ascend910b4" |
| config | dict | 是 | 配置字典 |
| device_pool | DevicePool | 否 | 设备池（已废弃，使用 WorkerManager） |
| workflow | str | 否 | 工作流名称（默认："default_workflow"） |
| inspirations | list | 否 | 来自进化的代码灵感 |
| meta_prompts | str | 否 | LLM 的元提示 |
| handwrite_suggestions | list | 否 | 手写优化建议 |

## 配置变化

### 废弃的配置
```yaml
# 不再需要（LangGraphTask 会忽略此项）
workflow_config_path: "config/default_workflow.yaml"
```

### 新增配置
```yaml
# 可选，默认为 config/parser_config.yaml
# parser_config_path: "config/parser_config.yaml"

# 工作流和限制（优先级：Task 参数 > 配置 > 默认值）
default_workflow: "default_workflow"  # 默认工作流名称
max_step: 20  # 最大迭代步数
```

## 工作流执行流程

### 默认工作流
```
designer → coder → verifier → [成功] → finish
                        ↓ [失败]
                   conductor → coder（重复）
```

### 仅 Coder 工作流
```
coder → verifier → [成功] → finish
            ↓ [失败]
       conductor → coder（重复）
```

## 迁移指南

### 步骤 1：替换导入
```python
# 修改前
from ai_kernel_generator.core.task import Task

# 修改后
from ai_kernel_generator.core.langgraph_task import LangGraphTask
```

### 步骤 2：替换实例化
```python
# 修改前
task = Task(...)

# 修改后
task = LangGraphTask(...)  # API 完全相同
```

### 步骤 3：更新 Worker 注册（可选但推荐）
```python
# 修改前（已废弃但仍可用）
device_pool = DevicePool([0, 1, 2, 3])
task = LangGraphTask(..., device_pool=device_pool)

# 修改后（新的服务化方式）
await register_local_worker([0, 1, 2, 3], backend='cuda', arch='a100')
task = LangGraphTask(..., device_pool=None)
```

## 文件映射

### 新增文件
| 文件 | 描述 |
|------|------|
| `core/langgraph_task.py` | 新的 Task 类，替代 `task.py` |
| `utils/langgraph/state.py` | 状态定义（`KernelGenState`） |
| `utils/langgraph/nodes.py` | 节点工厂（将 Agent 包装为节点，包含 Conductor 节点） |
| `utils/langgraph/routers.py` | 路由函数（条件边逻辑） |
| `workflows/base_workflow.py` | 工作流基类 |
| `workflows/default_workflow.py` | 默认工作流 |
| `workflows/coder_only_workflow.py` | 仅 Coder 工作流 |
| `workflows/verifier_only_workflow.py` | 仅 Verifier 工作流 |
| `workflows/connect_all_workflow.py` | 全连接工作流 |
| `utils/parser_loader.py` | Parser 加载器 |
| `config/parser_config.yaml` | Agent Parser 配置 |

### 已修改文件
| 文件 | 变化 |
|------|------|
| `core/evolve.py` | 导入并使用 `LangGraphTask` |
| `utils/evolve/evolution_processors.py` | 导入并使用 `LangGraphTask` |
| `core/agent/designer.py` | 支持 `parser_config_path`，Hint 模式输出转换 |
| `core/agent/coder.py` | 支持 `parser_config_path` |
| `config/*.yaml` | 移除 `workflow_config_path`，添加 `default_workflow` 和 `max_step` |

### 保留文件（向后兼容）
| 文件 | 状态 | 说明 |
|------|------|------|
| `core/task.py` | 保留 | 原 Task 类，部分测试仍在使用 |
| `core/agent/conductor.py` | 保留 | 原 Conductor 类，旧 Task 使用 |
| `config/*_workflow.yaml` | 保留 | 旧 Task 使用，LangGraphTask 忽略 |
| `utils/workflow_manager.py` | 保留 | 旧 Task 使用 |
| `utils/workflow_controller.py` | 保留 | 仅旧 Conductor 使用 |

## Conductor 分析

### 触发条件
- **Verifier 失败**：执行 Conductor 分析
- **Verifier 成功**：跳过 Conductor，直接结束

### Conductor 节点
Conductor 节点执行基于 LLM 的错误分析：
1. 加载 `conductor/analyze.j2` 模板
2. 准备包含错误信息的提示
3. 调用 LLM 进行分析
4. 解析决策和建议
5. 保存到 trace 和文件
6. 返回更新的状态

### 文件输出
- 决策和建议保存到 `{log_dir}/{op_name}/I{task_id}_S{step:02d}_conductor/`
- 格式与原 Conductor 实现相同

## 自定义工作流

### 创建自定义工作流
```python
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from langgraph.graph import StateGraph, START, END

class MyWorkflow(BaseWorkflow):
    def build_graph(self):
        workflow = StateGraph(KernelGenState)
        
        # 创建节点
        designer_node = NodeFactory.create_designer_node(
            self.agents['designer'], self.trace, self.config
        )
        coder_node = NodeFactory.create_coder_node(
            self.agents['coder'], self.trace
        )
        
        # 添加节点
        workflow.add_node("designer", designer_node)
        workflow.add_node("coder", coder_node)
        
        # 定义流程
        workflow.add_edge(START, "designer")
        workflow.add_edge("designer", "coder")
        workflow.add_edge("coder", END)
        
        return workflow.compile()
```

### 注册工作流
在 `langgraph_task.py` 中添加：
```python
WORKFLOW_REGISTRY = {
    "default": DefaultWorkflow,
    "my_workflow": MyWorkflow,  # 在此添加
    # ...
}
```