# AI Kernel Generator 运行示例教程

## 概述

本教程将介绍如何使用 AI Kernel Generator 运行单个算子的生成和验证流程。以 `examples/run_mindspore_triton_single.py` 为例，演示如何从 MindSpore 算子定义自动生成 Triton 实现并进行验证。

## 核心组件

### 主要模块
- **Task**: 任务实例
- **TaskPool**: 任务池管理器，支持异步任务执行
- **DevicePool**: 设备资源池，管理设备运行分配

### 执行流程
```
任务初始化 → AULDesigner生成AUL代码 → TritonCoder转换为Triton代码 → Verifier验证结果
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
- **输入生成**: `get_inputs()` 函数生成测试数据
- **初始化输入**: `get_init_inputs()` 函数提供模型初始化参数

### 2. 主执行函数

```python
async def run_mindspore_triton_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config()

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        impl_type="triton",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        device_pool=device_pool,
        framework="mindspore"
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
| impl_type | str | 实现类型："triton" 或 "swft" |
| backend | str | 计算后端："ascend"、"cuda"等 |
| arch | str | 硬件架构：如 "ascend910b4" 等 |
| config | dict | 配置文件，包含LLM模型配置等 |
| device_pool | DevicePool | 设备资源池 |
| framework | str | 框架类型："mindspore"、"torch" 或 "numpy" |

## 运行步骤

### 1. 环境准备

请参考项目[README文档](../README.md)

### 2. 执行示例

```bash
python examples/run_mindspore_triton_single.py
```
