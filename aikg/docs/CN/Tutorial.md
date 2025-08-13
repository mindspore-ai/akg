# AI Kernel Generator 运行示例教程

## 概述

本教程基于 coder-only 工作流，演示如何从 MindSpore 前端描述直接生成 Triton 实现并进行验证（不包含 Designer 步骤）。示例参考 `examples/run_mindspore_triton_single.py`。

## 核心组件

### 主要模块
- **Task**: 任务实例
- **TaskPool**: 任务池管理器，支持异步任务执行
- **DevicePool**: 设备资源池，管理设备运行分配

### 执行流程
```
任务初始化 → Coder 生成 Triton 代码 → Verifier 验证结果
```

## 示例代码解析

### 1. 任务描述函数

```python
def get_task_desc():
    return '''
import mindspore as ms
from mindspore import nn


class Model(nn.Cell):
    def __init__(self):
        super(Model, self).__init__()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return ms.ops.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = ms.ops.randn(batch_size, dim, dtype=ms.float16)
    return [x]


def get_init_inputs():
    return []
'''
```

**关键要素**：
- **模型定义**: 继承 `nn.Cell` 的 MindSpore 模型类
- **算子实现**: 在 `construct` 方法中定义具体的算子逻辑
- **输入生成**: `get_inputs()` 生成测试数据
- **初始化输入**: `get_init_inputs()` 提供模型初始化参数

### 2. 主执行函数（coder-only 工作流）

```python
async def run_mindspore_triton_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config(dsl="triton")  # 基于 DSL 选择默认方案

    # 推荐：运行前进行环境检查
    check_env_for_task("mindspore", "ascend", "triton", config)

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        device_pool=device_pool,
        framework="mindspore",
        workflow="coder_only_workflow"  # 关键：使用 coder-only 工作流
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    # 处理结果...
```

**参数说明**：
| 参数名称 | 类型 | 说明 |
|---------|------|------|
| op_name | str | 算子名称 |
| task_desc | str | 任务描述（包含模型定义和测试数据生成） |
| task_id | str | 任务唯一标识符 |
| dsl | str | 目标 DSL：如 "triton"、"swft" |
| backend | str | 计算后端：如 "ascend"、"cuda" |
| arch | str | 硬件架构：如 "ascend910b4" |
| config | dict | 任务编排方案配置（包含 `agent_model_config`、`workflow_config_path`、`docs_dir` 等） |
| device_pool | DevicePool | 设备资源池 |
| framework | str | 前端框架：如 "mindspore"、"torch"、"numpy" |
| workflow | str | 可选。覆盖配置中的 `workflow_config_path`，如 "coder_only_workflow" |

> 配置说明：`load_config("triton")` 会加载 `config/default_triton_config.yaml` 作为方案；若需 vLLM 本地推理且 coder-only，可使用 `vllm_triton_coderonly_config.yaml` 并通过 `load_config(config_path=...)` 显式指定。

## 运行步骤

### 1. 环境准备

请参考项目 [README 文档](../README_CN.md)。

### 2. 执行示例

```bash
python examples/run_mindspore_triton_single.py
```
