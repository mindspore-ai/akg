# AKG Agents 运行示例教程

## 概述

本教程演示如何使用 AKG Agents 的 LangGraph 任务系统生成高性能内核代码。基于 `LangGraphTask`（新架构），展示完整的代码生成与验证流程。

## 核心组件

### 主要模块
- **LangGraphTask**：任务执行器，基于 LangGraph 工作流引擎
- **TaskPool**：任务池管理器，支持异步批量任务执行
- **WorkerManager**：Worker 服务管理，负责远程/本地设备调度

### 执行流程
```
任务初始化 → 工作流选择 → Agent 执行（设计/编码/验证）→ 结果输出
```

## 示例 1：PyTorch Triton 单任务（CUDA）

参考 `examples/run_torch_triton_single.py`。

### 1. 任务描述函数

```python
def get_task_desc():
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]


def get_init_inputs():
    return []
'''
```

**关键要素**：
- **模型定义**：继承 `nn.Module` 的 PyTorch 模型类
- **算子实现**：在 `forward` 方法中定义具体的算子逻辑
- **输入生成**：`get_inputs()` 生成测试数据（需指定 device）
- **初始化输入**：`get_init_inputs()` 提供模型初始化参数

### 2. 主执行函数

```python
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.core.task_pool import TaskPool
from akg_agents.config.config_validator import load_config, check_env_for_task

async def run_torch_triton_single():
    op_name = "akg_relu"
    task_desc = get_task_desc()

    task_pool = TaskPool()

    # 注册本地 Worker（指定 GPU 设备列表）
    await register_local_worker([0], backend="cuda", arch="a100")

    # 加载配置
    config = load_config(dsl="triton_cuda", backend="cuda")

    # 推荐：运行前进行环境检查
    check_env_for_task("torch", "cuda", "triton_cuda", config)

    # 创建 LangGraph 任务
    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton_cuda",
        backend="cuda",
        arch="a100",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"  # 工作流选择
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()

    for op_name, success, task_info in results:
        if success:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")
```

### 参数说明

| 参数名称 | 类型 | 说明 |
|---------|------|------|
| op_name | str | 算子名称 |
| task_desc | str | 任务描述（包含模型定义和测试数据生成） |
| task_id | str | 任务唯一标识符 |
| dsl | str | 目标 DSL：`triton_cuda`、`triton_ascend` 等 |
| backend | str | 计算后端：`cuda`、`ascend`、`cpu` |
| arch | str | 硬件架构：`a100`、`ascend910b4` 等 |
| config | dict | 任务配置（通过 `load_config` 加载） |
| framework | str | 前端框架：`torch`、`mindspore`、`numpy` |
| workflow | str | 工作流名称（默认：`default`），可选：`coder_only`、`kernelgen_only`、`verifier_only` |

## 示例 2：Ascend NPU Triton 单任务

参考 `examples/run_torch_npu_triton_single.py`。

```python
async def run_torch_npu_triton_single():
    op_name = "akg_relu"
    task_desc = get_task_desc()  # 使用 npu tensor 的任务描述

    task_pool = TaskPool()

    # 注册 Ascend 设备
    await register_local_worker([0], backend="ascend", arch="ascend910b4")

    config = load_config(dsl="triton_ascend", backend="ascend")
    check_env_for_task("torch", "ascend", "triton_ascend", config)

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
```

## 示例 3：CPU C++ 单任务

参考 `examples/run_torch_cpu_cpp_single.py`。

```python
async def run_torch_cpu_cpp_single():
    op_name = "akg_relu"
    task_desc = get_task_desc()  # CPU 版本任务描述

    task_pool = TaskPool()

    # 注册 CPU Worker
    await register_local_worker([0], backend="cpu", arch="x86_64")

    config = load_config("cpp")
    check_env_for_task("torch", "cpu", "cpp", config)

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
```

## 可用工作流

| 工作流 | 流程 | 适用场景 |
|--------|------|----------|
| `default` | Designer → Coder ↔ Verifier | 完整流程：设计 → 编码 → 验证 |
| `coder_only` | Coder ↔ Verifier | 跳过设计，直接生成代码 |
| `kernelgen_only` | KernelGen ↔ Verifier | 基于 Skill 系统的代码生成 |
| `verifier_only` | Verifier → END | 仅验证已有代码 |
| `connect_all` | All ↔ All | 全连接，最大灵活性 |

## 工作流可视化

```python
# 创建任务后，可以可视化工作流
task = LangGraphTask(...)

# 打印 Mermaid 流程图
print(task.visualize())

# 保存为 PNG 文件
task.visualize(output_path="workflow.png")
```

## 运行步骤

### 1. 环境准备

请参考项目 [README 文档](../README_CN.md)。

### 2. 配置 LLM

通过环境变量或 `settings.json` 配置 LLM：
```bash
# 环境变量方式
export AKG_AGENTS_LLM_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_LLM_API_KEY="sk-xxx"
export AKG_AGENTS_LLM_MODEL_NAME="gpt-4"
```

或在 `.akg/settings.json` 中配置：
```json
{
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxx",
    "model_name": "gpt-4"
  }
}
```

### 3. 执行示例

```bash
# CUDA A100
python examples/run_torch_triton_single.py

# Ascend 910B
python examples/run_torch_npu_triton_single.py

# CPU C++
python examples/run_torch_cpu_cpp_single.py
```

## 相关文档
- [Workflow 系统文档](./Workflow.md)
- [KernelGen Agent 文档](./KernelGen.md)
- [KernelDesigner Agent 文档](./KernelDesigner.md)
- [AKG CLI 文档](./AKG_CLI.md)
