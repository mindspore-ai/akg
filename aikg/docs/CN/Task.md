# Task 模块设计文档

## 概述
Task 模块是 AI Kernel Generator 的核心组件，负责执行单个算子的编码、验证与优化流程。它集成了 Coder、Verifier 与 Conductor 等核心组件，并支持通过工作流灵活启用/禁用 Designer。在 coder-only 工作流下，不包含 Designer 步骤，直接由 Coder 生成目标 DSL（如 Triton）代码并进行验证。


## 核心功能
- **任务生命周期管理**：负责从初始化到验证完成的完整执行流程
- **多组件协调**：集成 Coder/Verifier/Conductor（可选 Designer）核心组件
- **硬件资源调度**：通过 `DevicePool` 实现 Ascend/NVIDIA 设备的分配与回收
- **执行控制**：通过工作流文件（`workflow.yaml`）中的 `limitation_info.required.max_step` 控制最大迭代步数


## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称（如 `matmul`） |
| task_desc | str (必选) | 任务描述，仅支持 mindspore/torch/numpy 的实现 |
| task_id | str (必选) | 任务 id，用于区分同名算子的不同 shape/计算 |
| backend | str (必选) | 计算后端：`ascend`/`cuda`/`cpu` |
| arch | str (必选) | 硬件架构，如 `ascend910b4`/`a100` 等 |
| dsl | str (必选) | 目标实现 DSL：`triton`/`swft`等 |
| config | dict (必选) | 任务编排方案配置，包含 `agent_model_config`、`workflow_config_path`、`docs_dir` 等 |
| device_pool | DevicePool (必选) | 设备资源池 |
| framework | str (必选) | 框架类型：`mindspore`/`torch`/`numpy` |
| task_type | str (可选) | 任务类型：`precision_only`（验证正确性）、`profile`（性能分析），默认 `precision_only` |
| workflow | str (可选) | 覆盖配置中的 `workflow_config_path`，如 `coder_only_workflow` |


## 执行流程 run

1. **初始化阶段**
   - 初始化 Conductor 控制模块
   - 按工作流启用所需 Agent（coder-only 流程不启用 Designer）
   - 配置 Coder 代码模板
   - 准备 Verifier 验证环境

2. **核心执行阶段**
   - 基于 Conductor 决策确定执行步骤
     - `coder`：调用 Coder 进行 Triton/SWFT 代码生成
     - `verifier`：调用 Verifier 进行精度/性能验证，并释放设备资源

3. **日志跟踪与迭代**
   - 将日志添加到Conductor的日志队列
   - 循环执行，直到达到最大迭代步数或验证通过

## 使用示例
```python
# 创建任务实例
task = Task(
    op_name="matmul",
    task_desc="...",
    backend="ascend",
    arch="ascend310p3",
    dsl="swft",
    config=load_config(dsl="triton"),
    device_pool=global_device_pool
)

# 执行任务
async def run_task():
    success = await task.run()
```