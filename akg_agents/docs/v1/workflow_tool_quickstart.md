# Workflow 工具化快速开始

## 简介

所有 Workflow 现在都可以作为工具被 KernelAgent 调用，LLM 会根据任务需求自动选择使用。

**已支持的 Workflow**:
- **CoderOnlyWorkflow** (`use_coder_only_workflow`): 跳过设计，直接生成+验证
- **DefaultWorkflow** (`use_default_workflow`): 完整流程（设计→生成→验证）
- **VerifierOnlyWorkflow** (`use_verifier_only_workflow`): 仅验证已有代码
- **ConnectAllWorkflow** (`use_connect_all_workflow`): 灵活流程，AI 智能决策

## 快速验证

### 1. 测试 Workflow 注册

```bash
cd akg_agents
python examples/test_coder_only_workflow_tool.py
```

预期输出：
```
✓ 已注册的 workflows (4): ['CoderOnlyWorkflow', 'DefaultWorkflow', 'VerifierOnlyWorkflow', 'ConnectAllWorkflow']
✓ CoderOnlyWorkflow → use_coder_only_workflow
✓ DefaultWorkflow → use_default_workflow
✓ VerifierOnlyWorkflow → use_verifier_only_workflow
✓ ConnectAllWorkflow → use_connect_all_workflow
✓ Workflow registry 大小: 4
🎉 所有测试通过！
```

### 2. 查看可用工具

```python
from akg_agents.op.agents.kernel_agent import KernelAgent

agent = KernelAgent(
    task_id="test",
    framework="torch",
    backend="cpu",
    arch="x86_64",
    dsl="cpp"
)

# 查看所有工具
for tool in agent.available_tools:
    name = tool["function"]["name"]
    desc = tool["function"]["description"][:50]
    print(f"- {name}: {desc}...")
```

你应该能看到所有 workflow 工具：
- `use_coder_only_workflow`
- `use_default_workflow`
- `use_verifier_only_workflow`
- `use_connect_all_workflow`

## 使用方式

### 方式 1: 自然语言（推荐）

让 LLM 自动选择：

```python
import asyncio
from akg_agents.op.agents.kernel_agent import KernelAgent

async def main():
    agent = KernelAgent(
        task_id="demo",
        framework="torch",
        backend="cpu",
        arch="x86_64",
        dsl="cpp"
    )
    
    # LLM 会根据需求自动选择工具
    result = await agent.run(
        "帮我完整开发一个 ReLU kernel，包括代码生成和验证"
    )
    
    print(f"状态: {result['status']}")
    print(f"输出: {result['output'][:200]}...")

asyncio.run(main())
```

### 方式 2: 直接调用 ToolExecutor

如果你想直接调用 workflow：

```python
from akg_agents.core_v2.tools.tool_executor import ToolExecutor

tool_executor = ToolExecutor(
    workflow_registry=agent.workflow_registry,
    agent_context={
        "task_id": "test",
        "framework": "torch",
        "backend": "cpu",
        "arch": "x86_64",
        "dsl": "cpp",
        "get_workflow_resources": lambda: agent._get_workflow_resources()
    }
)

result = await tool_executor.execute(
    "use_coder_only_workflow",
    {
        "op_name": "relu",
        "task_desc": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
        """,
        "dsl": "cpp",
        "framework": "torch",
        "backend": "cpu",
        "arch": "x86_64"
    }
)

print(result)
```

## LLM 何时会选择各个 Workflow？

根据 `DESCRIPTION` 中的说明，LLM 会根据任务特点选择不同的 workflow：

### CoderOnlyWorkflow
✅ **适用场景**:
- "完整开发一个 ReLU kernel"
- "生成并验证 MatMul 代码"
- 需求明确，不需要设计阶段

### DefaultWorkflow
✅ **适用场景**:
- "设计并实现一个高性能的 Attention"
- "开发一个优化的 LayerNorm，需要设计优化方案"
- 任务复杂，需要设计阶段

### VerifierOnlyWorkflow
✅ **适用场景**:
- "验证这段代码的正确性"
- "测试我写的 kernel 性能"
- 已有代码，只需要验证

### ConnectAllWorkflow
✅ **适用场景**:
- "极其复杂的算子开发"
- "需要最大灵活性"
- 标准流程无法满足时的备选方案

❌ **不会选择 Workflow**（会用 SubAgent）:
- "快速生成一个 ReLU kernel"（用 `call_kernel_gen`）
- "给我一个代码草稿"（用 `call_kernel_gen`）
- "设计一个算法方案"（用 `call_kernel_designer`）

## Workflow vs SubAgent

| 特性 | CoderOnly | Default | VerifierOnly | ConnectAll | SubAgent |
|------|-----------|---------|--------------|------------|----------|
| 执行时间 | 1-5min | 2-8min | 0.5-2min | 2-10min | 10-30s |
| 包含设计 | ❌ | ✅ | ❌ | ✅ | ❌ |
| 包含验证 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 迭代优化 | ✅ | ✅ | ❌ | ✅ | ❌ |
| 灵活性 | 固定 | 固定 | 固定 | 最高 | N/A |
| 适用场景 | 简单完整开发 | 复杂完整开发 | 仅验证 | 极复杂任务 | 快速原型 |

## 常见问题

### Q1: Workflow 执行很慢？

A: 正常。Workflow 包含完整的代码生成、验证、迭代流程，通常需要 1-5 分钟。如果只需要代码草稿，使用 `call_kernel_gen`。

### Q2: 如何查看 Workflow 内部执行状态？

A: 目前 Workflow 执行是黑盒的，只能在最后看到结果。后续会支持流式输出。

### Q3: Workflow 失败了怎么办？

A: 查看 `error_information` 字段了解失败原因。常见原因：
- 设备不可用（需要 GPU/CPU）
- 框架代码格式错误
- 验证超时

### Q4: 如何禁用 Workflow 工具？

A: 修改 `kernel_agent.py` 的 `_load_workflow_registry()` 方法，注释掉相应的导入：

```python
def _load_workflow_registry(self):
    # 注释掉这行就不会加载 workflow
    # from akg_agents.op.workflows import coder_only_workflow
    return {}
```

## 下一步

- 查看完整设计文档: `docs/workflow_as_tool.md`
- 添加更多 Workflow: 参考 CoderOnlyWorkflow 的实现
- 运行集成测试: `pytest tests/v2/ut/test_workflow_registry.py`
