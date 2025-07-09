# Task 模块设计文档

## 概述
Task模块是AI Kernel Generator的核心组件，负责执行单个算子的设计、编码、验证和优化流程。它集成了Designer、Coder、Verifier和Conductor四个核心组件，实现了从AUL代码到Triton/SWFT代码的完整转换和验证过程。


## 核心功能
- **任务生命周期管理**：负责从初始化到验证完成的完整执行流程
- **多组件协调**：集成Designer/Coder/Verifier/Conductor四大核心组件
- **硬件资源调度**：通过DevicePool实现Ascend/NVIDIA设备的分配与回收
- **执行控制**：通过limit_steps参数控制最大迭代步数


## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称（如"matmul"） |
| task_desc | str (必选) | 任务描述，仅支持mindspore/torch/numpy的实现 |
| task_id | str (必选) | 任务id，用于区分op_name相同时不同shape或者计算的标识符 |
| backend | str (必选) | 计算后端，仅支持ascend/cuda/cpu |
| arch | str (必选) | 硬件架构，根据backend的不同，硬件架构也不相同，如ascend910b4/a100等 |
| impl_type | str (必选) | 后端实现类型，仅triton和swft |
| config | dict (必选) | 包含LLM配置/日志路径等 |
| device_pool | DevicePool (必选) | 设备资源池 |
| framework | str (必选) | 框架类型，仅支持mindspore/torch/numpy |
| task_type | str (可选) | 任务类型，仅支持precision_only（验证结果正确性）, profile（用于性能分析）， 默认：precision_only |
| limit_steps | int (可选) | 最大执行步数限制，默认：10 |


## 执行流程 run

1. **初始化阶段**
   - 初始化Conductor控制模块
   - 加载Designer配置参数
   - 配置Coder代码模板
   - 准备Verifier验证环境

2. **核心执行阶段**
   - 基于Conductor决策的 action_type 确定执行步骤
     - `designer`：调用Designer进行AUL代码生成
     - `coder`：调用Coder进行Triton/SWFT代码转换
     - `verifier`：调用Verifier进行精度/性能验证，并释放设备资源

3. **日志跟踪与迭代**
   - 将日志添加到Conductor的日志队列
   - 循环执行，直到达到最大迭代步数或验证通过

## 使用示例
```python
# 创建任务实例
task = Task(
    op_name="swish",
    task_desc="Swish激活函数: x * sigmoid(beta * x)",
    backend="ascend",
    arch="ascend310p3",
    impl_type="swft",
    config=load_config(),
    device_pool=global_device_pool
)

# 执行任务
async def run_task():
    success = await task.run()
    print(f"Task completed: {success}")
```