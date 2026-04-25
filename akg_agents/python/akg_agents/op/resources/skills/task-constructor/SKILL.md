---
name: task-constructor
description: >
  从 PyTorch/Triton 代码仓中提取算子实现，构建为 KernelBench 格式的标准化单文件自包含任务。
  支持代码提取、AST 依赖追踪、函数内联、import 清理、格式验证和参考对比测试。
  当用户需要从现有代码构建 task_code 时使用此 Skill。
category: workflow
version: "1.0.0"
metadata:
  task_type: code_transformation
  target_format: kernelbench
  input_types: code,file,directory
---

# 标准化任务构建工作流

## 适用场景

- 用户提供 PyTorch/Triton 代码仓路径，需要提取算子并构建 task_code
- 用户提供代码片段，需要包装成 KernelBench 格式的标准化任务
- 用户指定某个 torch 内部函数，需要提取其分解实现

## 目标格式

最终生成的文件必须是 **单一自包含 Python 文件**：

```python
import torch
import torch.nn as nn

# 所有依赖函数内联（不能 import 外部文件）

class Model(nn.Module):
    def __init__(self, <params>):
        super(Model, self).__init__()

    def forward(self, <inputs>) -> torch.Tensor:
        return output

def get_inputs():
    return [input1, input2, ...]

def get_init_inputs():
    return [param1, ...]
```

## 工具使用指南

### 调用 `call_task_constructor`

此工具内部运行完整的 ReAct 循环，自动完成以下步骤：

1. **定位目标代码**：搜索目标函数
2. **依赖追踪**：AST 分析自动发现所有依赖（同文件函数 + 外部模块调用）
3. **任务装配**：选择最佳策略（排除式/选择性/完整嵌入）构建自包含文件
4. **验证**：格式验证（实例化 + forward + NaN/Inf + 一致性检查）
5. **参考对比**：与原始 torch 函数对比多组输入

### 参数

- `user_input`：用户需求描述（如 "从 pytorch 仓中提取 xxx 的分解实现"）
- `source_path`：可选，代码仓/文件路径

### 返回

- `task_code`：生成的标准化任务代码
- `task_code_path`：代码文件路径
- `op_name`：算子名称
- `summary`：构建过程摘要

## 核心规则

1. **禁止重写复杂函数**：原始函数可运行就直接复用，一行都不改
2. **返回值必须一致**：多张量返回就返回 tuple，不能截断
3. **内联外部函数前先查签名**：通过依赖追踪自动检测外部调用来源模块

## Scripts

1. `scripts/validate_kernelbench_task.py` - 验证 task 代码是否符合 KernelBench 格式（参数：`--stdin --json`）

### 使用示例

```
Think: 需要验证生成的 task 代码是否正确
Action: execute_script(script_path="resources/skills/task-constructor/scripts/validate_kernelbench_task.py", args="--stdin --json", stdin_input="<task 代码>")
Observation: {"valid": true, "static_check": {...}, "runtime_check": {...}}
```

也可以直接验证文件：

```
Action: execute_script(script_path="resources/skills/task-constructor/scripts/validate_kernelbench_task.py", args="/path/to/task.py --json")
```

## 参考文档

- `references/kernelbench-format.md` - 格式规范
- `references/assembly-strategies.md` - 装配策略说明
