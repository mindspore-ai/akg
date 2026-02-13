# Workflow 工具化设计文档

## 概述

将 LangGraph Workflow 封装为可调用的工具，使其能够被 KernelAgent 在 ReAct 循环中动态选择和调用。

## 设计原则

1. **职责分离**：Workflow 和 Agent 是不同的概念，使用独立的注册机制
2. **统一接口**：Workflow 作为工具，与其他工具（basic_tool, agent_tool）使用相同的调用接口
3. **保持兼容**：不破坏现有的 Workflow 实现和继承关系
4. **易于扩展**：新增 Workflow 只需添加装饰器和元数据

## 架构设计

### 1. WorkflowRegistry（Workflow 注册中心）

类似于 `AgentRegistry`，但专门用于 Workflow。

**位置**: `akg_agents/python/akg_agents/core_v2/workflows/registry.py`

**功能**:
- 注册 Workflow 类
- 发现已注册的 Workflow
- 提取工具配置元数据
- 支持应用范围（scope）隔离

**使用示例**:
```python
from akg_agents.core_v2.workflows.registry import register_workflow

@register_workflow(scopes=["op"])
class CoderOnlyWorkflow(OpBaseWorkflow):
    # 工具配置元数据
    TOOL_NAME = "use_coder_only_workflow"
    DESCRIPTION = "使用 CoderOnly workflow 生成代码"
    PARAMETERS_SCHEMA = {...}
    
    # 原有的 workflow 实现
    def build_graph(self):
        ...
```

### 2. Workflow 工具配置元数据

每个 Workflow 需要定义 3 个类属性：

| 属性 | 类型 | 说明 |
|-----|------|------|
| `TOOL_NAME` | str | 工具名称，如 "use_coder_only_workflow" |
| `DESCRIPTION` | str | 功能描述，用于 LLM 理解何时使用此 workflow |
| `PARAMETERS_SCHEMA` | dict | 参数 schema（JSON Schema 格式） |

**示例**:
```python
TOOL_NAME = "use_coder_only_workflow"

DESCRIPTION = """
使用 CoderOnly workflow 生成 kernel 代码（跳过设计阶段）。

完整流程：
1. Coder: 根据任务描述生成代码
2. Verifier: 验证正确性和性能
3. Conductor: 分析失败原因并指导修复（如果验证失败）

适用场景：
- 需求明确，无需额外设计阶段
- 需要完整的代码生成、验证、迭代流程

注意：执行时间较长（1-5分钟）
"""

PARAMETERS_SCHEMA = {
    "type": "object",
    "properties": {
        "op_name": {
            "type": "string",
            "description": "算子名称"
        },
        "task_desc": {
            "type": "string",
            "description": "任务描述（框架代码）"
        },
        ...
    },
    "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]
}
```

### 3. KernelAgent 集成

**修改点**:
1. 添加 `_load_workflow_registry()` 方法，加载已注册的 workflow
2. 添加 `_get_workflow_resources()` 方法，延迟初始化 workflow 所需资源
3. 将 workflow 工具添加到 `available_tools`

**流程**:
```
KernelAgent.__init__()
  ├─ _load_available_tools()      # 加载基础工具
  ├─ _load_agent_registry()       # 加载 agent 工具
  ├─ _load_workflow_registry()    # 加载 workflow 工具（新增）
  └─ ToolExecutor(
        agent_registry,
        workflow_registry,          # 传递 workflow registry（新增）
        agent_context={
            "get_workflow_resources": lambda: self._get_workflow_resources()
        }
     )
```

### 4. ToolExecutor 扩展

**修改点**:
1. 添加 `workflow_registry` 参数
2. 在 `execute()` 方法中检查 workflow
3. 添加 `_execute_workflow()` 方法

**执行流程**:
```
ToolExecutor.execute(tool_name, arguments)
  ├─ if tool_name in agent_registry:
  │     └─ _execute_agent()
  ├─ if tool_name in workflow_registry:    # 新增
  │     └─ _execute_workflow()             # 新增
  └─ else:
        └─ _execute_basic_tool() / _execute_domain_tool()
```

**Workflow 执行流程**:
```python
async def _execute_workflow(tool_name, arguments):
    # 1. 获取 workflow 类和资源
    workflow_class = workflow_registry[tool_name]["workflow_class"]
    workflow_resources = get_workflow_resources()  # 通过回调
    
    # 2. 创建和编译 workflow
    workflow = workflow_class(**workflow_resources)
    app = workflow.compile()
    
    # 3. 构建初始状态
    initial_state = _build_workflow_state(arguments)
    
    # 4. 执行 workflow
    final_state = await app.ainvoke(initial_state)
    
    # 5. 格式化结果
    return _format_workflow_result(final_state)
```

## 资源管理

Workflow 需要多种资源：
- **agents**: Designer, Coder, Verifier 实例
- **trace**: 跟踪系统
- **config**: 配置字典
- **device_pool / worker_manager**: 设备和 Worker 管理

**解决方案**: 延迟初始化 + 回调

```python
class KernelAgent:
    def _get_workflow_resources(self):
        """延迟初始化 workflow 资源"""
        if self._workflow_resources is None:
            agents = {
                "designer": Designer(...),
                "coder": Coder(...),
                "verifier": KernelVerifier(...)
            }
            self._workflow_resources = {
                "agents": agents,
                "trace": self.trace,
                "config": {},
                "backend": self.backend,
                "arch": self.arch
            }
        return self._workflow_resources
```

## 使用示例

### 1. LLM 视角

LLM 在 ReAct 循环中可以看到并选择 workflow 工具：

```json
{
  "available_tools": [
    {
      "name": "call_kernel_gen",
      "description": "快速生成 kernel 代码草稿..."
    },
    {
      "name": "use_coder_only_workflow",
      "description": "使用 CoderOnly workflow 生成代码（完整流程，包括验证）..."
    }
  ]
}
```

LLM 决策：
```json
{
  "tool_name": "use_coder_only_workflow",
  "arguments": {
    "op_name": "relu",
    "task_desc": "class Model(nn.Module): ...",
    "dsl": "triton",
    "framework": "torch",
    "backend": "cuda",
    "arch": "a100"
  },
  "reason": "用户要求完整开发并验证，使用 workflow 更合适"
}
```

### 2. 用户视角

用户可以自然地描述需求：

```python
# 场景 1: 快速生成代码草稿
await kernel_agent.run("帮我生成一个 ReLU kernel")
# LLM 可能选择: call_kernel_gen

# 场景 2: 完整开发流程
await kernel_agent.run("帮我完整开发一个 MatMul kernel，包括验证")
# LLM 可能选择: use_coder_only_workflow

# 场景 3: 只做设计
await kernel_agent.run("帮我设计一个 Softmax 的算法方案")
# LLM 可能选择: call_kernel_designer
```

## 添加新的 Workflow

### 步骤 1: 添加装饰器和元数据

```python
from akg_agents.core_v2.workflows.registry import register_workflow

@register_workflow(scopes=["op"])
class MyNewWorkflow(OpBaseWorkflow):
    """自定义 Workflow"""
    
    # 工具配置元数据
    TOOL_NAME = "use_my_new_workflow"
    DESCRIPTION = "描述何时使用此 workflow..."
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            ...
        },
        "required": [...]
    }
    
    # Workflow 实现
    def build_graph(self):
        workflow = StateGraph(KernelGenState)
        # ... 构建图
        return workflow
```

### 步骤 2: 在 KernelAgent 中导入

修改 `kernel_agent.py`:
```python
def _load_workflow_registry(self):
    try:
        from akg_agents.op.workflows import (
            coder_only_workflow,
            my_new_workflow,  # 添加新的导入
        )
    except Exception as e:
        logger.warning(f"导入 workflows 失败: {e}")
```

### 步骤 3: 测试

```python
# 测试注册
from akg_agents.core_v2.workflows.registry import WorkflowRegistry
assert "MyNewWorkflow" in WorkflowRegistry.list_workflows()

# 测试工具配置
config = WorkflowRegistry.get_tool_config("MyNewWorkflow")
assert config is not None
```

## 测试

### 单元测试

```bash
# 测试 WorkflowRegistry
pytest tests/v2/ut/test_workflow_registry.py -v

# 测试 CoderOnlyWorkflow 注册
pytest tests/v2/ut/test_workflow_registry.py::TestCoderOnlyWorkflowIntegration -v
```

### 集成测试

```bash
# 测试完整集成
python examples/test_coder_only_workflow_tool.py
```

### 端到端测试

```bash
# 使用 KernelAgent 调用 workflow
python examples/run_kernel_agent.py --query "完整开发一个 ReLU kernel" --backend cpu --dsl cpp
```

## 当前支持的 Workflow

| Workflow | 工具名称 | 描述 | 状态 |
|---------|---------|------|------|
| CoderOnlyWorkflow | `use_coder_only_workflow` | 跳过设计，直接生成+验证 | ✅ 已实现 |
| DefaultWorkflow | `use_default_workflow` | Designer→Coder→Verifier 完整流程 | ✅ 已实现 |
| VerifierOnlyWorkflow | `use_verifier_only_workflow` | 仅验证已有代码 | ✅ 已实现 |
| ConnectAllWorkflow | `use_connect_all_workflow` | 全连接，AI 智能决策路由 | ✅ 已实现 |

## 优势与局限

### ✅ 优势

1. **灵活性**: LLM 可以根据任务复杂度动态选择 workflow 或 subagent
2. **统一性**: Workflow 和 Agent 使用相同的工具调用机制
3. **代码复用**: 充分利用现有的 Workflow 实现
4. **易于扩展**: 添加新 workflow 只需装饰器和元数据

### ⚠️ 局限

1. **执行时间**: Workflow 通常执行较长时间（1-5分钟）
2. **资源开销**: 需要初始化多个 Agent（Designer, Coder, Verifier）
3. **透明度**: Workflow 内部状态不如 Agent 调用链清晰
4. **错误处理**: Workflow 内部迭代的错误不易暴露

## 后续优化方向

1. **流式输出**: 支持 Workflow 内部节点的实时状态反馈
2. **断点续传**: 支持 Workflow 执行中断和恢复
3. **性能优化**: 缓存 Agent 实例，避免重复初始化
4. **更多 Workflow**: 支持 DefaultWorkflow, VerifierOnlyWorkflow 等

## 参考

- Agent 注册机制: `akg_agents/core_v2/agents/registry.py`
- Workflow 基类: `akg_agents/op/workflows/base_workflow.py`
- 工具执行器: `akg_agents/core_v2/tools/tool_executor.py`
