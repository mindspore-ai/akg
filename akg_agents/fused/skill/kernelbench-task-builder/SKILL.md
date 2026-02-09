---
name: kernelbench-task-builder
description: >
  从 PyTorch/Triton 代码仓中提取算子实现，构建为 KernelBench 标准格式的
  单文件自包含任务。支持代码提取、AST 依赖追踪、函数内联、import 清理、
  格式验证和参考对比测试。
level: L1
category: workflow
version: "1.0.0"
metadata:
  task_type: code_transformation
  target_format: kernelbench
  input_types: code,file,directory
---

# KernelBench 任务构建工作流

## 目标格式

最终生成的文件必须是**单一自包含 Python 文件**，格式如下：

```python
import torch
import torch.nn as nn

# ... 所有依赖函数（内联，不能 import 外部文件）...

class Model(nn.Module):
    def __init__(self, <params>):
        super(Model, self).__init__()

    def forward(self, <inputs>):
        return output  # 可以返回单个 Tensor 或 Tuple[Tensor, ...]

def get_inputs():
    return [input1, input2, ...]

def get_init_inputs():
    return [param1, ...]
```

## 工作流程

### 第一步：定位目标代码（1-3 步）

1. `grep_search` 搜索目标函数（优先于 scan_dir，更高效）
2. `copy_to_workspace` 复制源文件和 benchmark/test 文件
3. `read_function` 提取目标函数查看实现

### 第二步：依赖追踪（关键！）

4. **`trace_dependencies` 自动追踪函数依赖**
   - 输入: 文件路径 + 入口函数名
   - 输出: 同文件内所有依赖函数 + 需要内联的外部调用（附带来源模块路径）
   - 工具自动分析文件 import，通过私有模块检测判断哪些外部调用需要内联
   - 对于外部调用：按工具提示的来源路径用 `read_function` 查看原始签名后再内联

5. `read_function` 提取 benchmark 的 create_config/create_inputs（了解输入形状）

### 第三步：选择策略构建

根据依赖分析结果选择构建策略：

**策略 A: 排除式嵌入（首选，依赖大部分函数时）**
- `assemble_task` 的 `source_files` 用 `exclude_functions` 模式
- 嵌入整个文件但自动移除不需要的函数/类

**策略 A': 选择性提取（依赖少数函数时）**
- `assemble_task` 的 `source_files` 用 `functions` 列表
- 函数列表应来自 `trace_dependencies` 的结果
- 工具自动清除装饰器 + 精简未使用的 import

**策略 A'': 完整嵌入（不需要剔除任何函数时）**
- `source_files` 用字符串路径

**策略 B: 分段追加（需要修改源函数时）**
- `write_file` + `append_to_file` 逐段生成
- **追加的函数必须从 workspace 中复制原始代码，禁止自己重写！**

### 第四步：验证和完成

7. `validate_task` 验证（实例化 + forward + NaN/Inf + 一致性）
8. **`test_with_reference`** 对比原始函数验证正确性
   - 当原始函数存在于 torch 中时，直接调用它作为 reference
   - 构造多组输入覆盖边界情况（不同形状、不同 dim、不同 dtype 等）
9. `finish`

## 核心规则

### 规则 1: 禁止重写复杂函数
- 原始函数可运行 → 直接复用，一行都不要改
- 有 NPU/Triton 特定代码 → 检查是否有 fallback
- 确实需要修改 → 只改最小必要部分
- **一个 50 行的原始函数包含精确的索引计算。自己写的"简化版"几乎必定有 bug。**

### 规则 2: 返回值必须与原始函数完全一致
- 多张量返回 → forward 也必须返回 tuple
- **绝对不能擅自截断返回值**

### 规则 3: 不确定时必须 ask_user
- 不要自行假设格式要求、返回值数量、输入格式等

### 规则 4: 内联外部函数前先查签名
- `trace_dependencies` 会自动检测并标注来源模块路径
- 用 `read_function` 查看原始实现
- 参数签名必须与原始完全一致

## 参考文档

1. `references/kernelbench-format.md` - KernelBench 格式详细规范
2. `references/workflow-steps.md` - 工作流各步骤详细说明
