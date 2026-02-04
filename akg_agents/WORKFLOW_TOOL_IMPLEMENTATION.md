# Workflow 工具化实现总结

## 实现概述

将所有主要 Workflow 封装为可被 `KernelAgent` 调用的工具，使 LLM 能够在 ReAct 循环中动态选择使用快速的 SubAgent 还是完整的 Workflow，以及选择哪个 Workflow。

**已实现的 Workflows**:
1. **CoderOnlyWorkflow** - 跳过设计，直接生成+验证
2. **DefaultWorkflow** - 完整流程（设计→生成→验证）
3. **VerifierOnlyWorkflow** - 仅验证已有代码
4. **ConnectAllWorkflow** - 全连接，AI 智能决策路由

## 核心设计

- **独立注册**: 创建 `WorkflowRegistry` 和 `@register_workflow` 装饰器，独立于 Agent
- **统一接口**: Workflow 作为工具，与 Agent 工具使用相同的调用接口
- **延迟初始化**: Workflow 所需资源（agents, trace 等）延迟创建
- **保持兼容**: 不破坏现有 Workflow 的实现和继承关系

## 修改的文件

### 新增文件 (5 个)

1. **`python/akg_agents/core_v2/workflows/__init__.py`**
   - WorkflowRegistry 模块导出

2. **`python/akg_agents/core_v2/workflows/registry.py`** (237 行)
   - `WorkflowRegistry` 类：Workflow 注册中心
   - `@register_workflow` 装饰器
   - 工具配置元数据提取

3. **`tests/v2/ut/test_workflow_registry.py`** (115 行)
   - WorkflowRegistry 单元测试
   - CoderOnlyWorkflow 注册集成测试

4. **`examples/test_coder_only_workflow_tool.py`** (157 行)
   - Workflow 注册验证测试
   - KernelAgent 集成测试
   - 工具详细信息查看

5. **`docs/workflow_as_tool.md`** (详细设计文档)
   - 完整的设计文档
   - 架构说明
   - 使用示例和最佳实践

6. **`docs/workflow_tool_quickstart.md`** (快速开始指南)
   - 快速验证步骤
   - 使用示例
   - 常见问题

### 修改的文件 (7 个)

1. **`python/akg_agents/op/workflows/coder_only_workflow.py`**
   
   **修改内容**:
   - 添加导入: `from akg_agents.core_v2.workflows.registry import register_workflow`
   - 添加装饰器: `@register_workflow(scopes=["op"])`
   - 添加工具配置元数据:
     - `TOOL_NAME = "use_coder_only_workflow"`
     - `DESCRIPTION = """..."""`  (详细描述何时使用)
     - `PARAMETERS_SCHEMA = {...}`  (参数定义)

2. **`python/akg_agents/op/workflows/default_workflow.py`**
   
   **修改内容**:
   - 添加导入和装饰器
   - 添加工具配置元数据:
     - `TOOL_NAME = "use_default_workflow"`
     - `DESCRIPTION`: 强调包含设计阶段，适合复杂任务
     - `PARAMETERS_SCHEMA`: 参数定义

3. **`python/akg_agents/op/workflows/verifier_only_workflow.py`**
   
   **修改内容**:
   - 添加导入和装饰器
   - 添加工具配置元数据:
     - `TOOL_NAME = "use_verifier_only_workflow"`
     - `DESCRIPTION`: 强调仅验证，需要提供代码
     - `PARAMETERS_SCHEMA`: 增加 `generated_code` 参数

4. **`python/akg_agents/op/workflows/connect_all_workflow.py`**
   
   **修改内容**:
   - 添加导入和装饰器
   - 添加工具配置元数据:
     - `TOOL_NAME = "use_connect_all_workflow"`
     - `DESCRIPTION`: 强调灵活性和智能决策
     - `PARAMETERS_SCHEMA`: 参数定义

5. **`python/akg_agents/core_v2/agents/kernel_agent.py`**
   
   **修改内容**:
   - 在 `__init__` 中添加:
     ```python
     self.workflow_registry = self._load_workflow_registry()
     self._workflow_resources = None
     ```
   - 修改 `ToolExecutor` 初始化，传递 `workflow_registry` 和资源回调
   - 新增方法 `_load_workflow_registry()`: 加载已注册的 workflow
   - 新增方法 `_get_workflow_resources()`: 延迟初始化 workflow 所需资源

6. **`python/akg_agents/core_v2/tools/tool_executor.py`**
   
   **修改内容**:
   - 在 `__init__` 中添加 `workflow_registry` 参数
   - 修改 `execute()` 方法，添加 workflow 检查分支
   - 新增方法 `_execute_workflow()`: 执行 workflow
   - 新增方法 `_build_workflow_state()`: 构建 workflow 初始状态
   - 新增方法 `_format_workflow_result()`: 格式化 workflow 结果

7. **`examples/test_coder_only_workflow_tool.py`**
   
   **修改内容**:
   - 更新测试以验证所有 4 个 workflow
   - 更新文档和输出信息

## 关键代码片段

### 1. Workflow 注册

```python
@register_workflow(scopes=["op"])
class CoderOnlyWorkflow(OpBaseWorkflow):
    TOOL_NAME = "use_coder_only_workflow"
    DESCRIPTION = """使用 CoderOnly workflow..."""
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
    
    def build_graph(self):
        # 原有实现不变
        ...
```

### 2. KernelAgent 加载

```python
class KernelAgent:
    def __init__(self, ...):
        self.workflow_registry = self._load_workflow_registry()
        self._workflow_resources = None
        
        self.tool_executor = ToolExecutor(
            workflow_registry=self.workflow_registry,
            agent_context={
                "get_workflow_resources": lambda: self._get_workflow_resources()
            }
        )
```

### 3. Workflow 执行

```python
class ToolExecutor:
    async def execute(self, tool_name, arguments):
        if tool_name in self.workflow_registry:
            return await self._execute_workflow(tool_name, arguments)
    
    async def _execute_workflow(self, tool_name, arguments):
        # 1. 获取资源
        workflow_resources = get_resources()
        
        # 2. 创建并编译
        workflow = workflow_class(**workflow_resources)
        app = workflow.compile()
        
        # 3. 执行
        final_state = await app.ainvoke(initial_state)
        
        return format_result(final_state)
```

## 测试验证

### 运行测试

```bash
# 单元测试
pytest tests/v2/ut/test_workflow_registry.py -v

# 集成测试
python examples/test_coder_only_workflow_tool.py
```

### 预期输出

```
✓ 已注册的 workflows (4): ['CoderOnlyWorkflow', 'DefaultWorkflow', 'VerifierOnlyWorkflow', 'ConnectAllWorkflow']
✓ CoderOnlyWorkflow → use_coder_only_workflow
✓ DefaultWorkflow → use_default_workflow
✓ VerifierOnlyWorkflow → use_verifier_only_workflow
✓ ConnectAllWorkflow → use_connect_all_workflow
✓ Workflow registry 大小: 4
🎉 所有测试通过！所有 Workflow 已成功集成为工具。
```

## 使用示例

```python
from akg_agents.core_v2.agents.kernel_agent import KernelAgent

agent = KernelAgent(
    task_id="demo",
    framework="torch",
    backend="cpu",
    arch="x86_64",
    dsl="cpp"
)

# LLM 会根据需求自动选择 workflow 或 subagent
result = await agent.run(
    "帮我完整开发一个 ReLU kernel，包括代码生成和验证"
)
```

## 已实现的 Workflows

| Workflow | 工具名称 | 说明 | 状态 |
|---------|---------|------|------|
| CoderOnlyWorkflow | `use_coder_only_workflow` | 跳过设计，直接生成+验证 | ✅ 已实现 |
| DefaultWorkflow | `use_default_workflow` | 完整流程：设计→生成→验证 | ✅ 已实现 |
| VerifierOnlyWorkflow | `use_verifier_only_workflow` | 仅验证已有代码 | ✅ 已实现 |
| ConnectAllWorkflow | `use_connect_all_workflow` | 全连接，AI 智能决策 | ✅ 已实现 |

### Workflow 选择逻辑

LLM 会根据任务特点自动选择合适的 workflow：

- **简单明确的任务** → `CoderOnlyWorkflow`
- **复杂需要设计的任务** → `DefaultWorkflow`
- **已有代码需要验证** → `VerifierOnlyWorkflow`
- **极其复杂需要灵活性** → `ConnectAllWorkflow`
- **快速原型** → `call_kernel_gen` (SubAgent)

## 文档

- 📖 完整设计文档: `docs/workflow_as_tool.md`
- 🚀 快速开始指南: `docs/workflow_tool_quickstart.md`

## 总结

- ✅ 创建独立的 WorkflowRegistry 和注册机制
- ✅ 修改 4 个 Workflow 添加工具配置元数据
  - CoderOnlyWorkflow
  - DefaultWorkflow
  - VerifierOnlyWorkflow
  - ConnectAllWorkflow
- ✅ 在 KernelAgent 中集成 workflow 加载
- ✅ 扩展 ToolExecutor 支持 workflow 执行
- ✅ 编写单元测试和集成测试
- ✅ 提供完整文档和使用示例

所有主要 Workflow 现在都可以作为工具被 KernelAgent 调用，LLM 会根据任务需求自动选择最合适的 workflow 或 subagent！
