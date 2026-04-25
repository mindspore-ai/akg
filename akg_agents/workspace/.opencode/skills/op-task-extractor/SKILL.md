---
name: op-task-extractor
description: >
  从用户提供的代码或自然语言描述，构建标准化的单文件自包含任务代码。
argument-hint: >
  需要提供：1) 用户代码文件路径或自然语言描述；2) 可选：shape/dtype 信息来源文件路径
---

## What I do

将用户的输入（代码文件或自然语言描述）转化为标准格式的 `{op_name}.py` 任务文件，并通过验证脚本确认可执行。

## When to use me

需要为后续算子生成工作流准备标准化任务文件时

## Workflow

```
Step 1  判断输入类型
  ├─ 用户提供了符合格式要求的代码文件 → 保存为 {op_name}.py → Step 4（验证）
  │     ├─ 通过 → 返回（任务结束）
  │     └─ 失败 → Step 2（分析并重写）
  └─ 用户提供自然语言描述 → Step 2
Step 2  代码分析 & 依赖追踪
Step 3  构建 {op_name}.py（参考末尾「输出格式」）
Step 4  运行验证脚本（必须执行）
Step 5  用户确认
```

### Step 1: 判断输入类型

- **用户提供了符合格式要求的代码文件** → 直接保存为 `{op_name}.py`，跳到 Step 4 运行验证脚本
- **其他** → 进入 Step 2

### Step 2: 代码分析 & 依赖追踪

- 读取用户确认的 `framework`, `backend`, `arch` 配置
- 如有源代码：
  - 识别待优化部分，提取 shape/dtype 信息，确定输入/输出签名
  - 分析依赖关系（AST 级别），追踪自定义函数/类，确定需要内联的外部依赖
- 如为自然语言描述：
  - 从描述中理解算子语义，确定合理的 shape/dtype 默认值

### Step 3: 构建 {op_name}.py

按末尾「输出格式」生成文件，用 PyTorch/Python 实现：

- 将算子逻辑包装到 `Model.forward()` 中
- 如有初始化状态（权重、参数），放入 `Model.__init__()`
- 将所有自定义依赖内联到文件中
- 根据 shape/dtype 信息构建 `get_inputs()` 和 `get_init_inputs()`
- 如用户未提供 shape/dtype，从代码上下文推断合理默认值

### Step 4: 运行验证脚本（必须执行）

使用命令模板执行 `@scripts/validate_kernelbench_task.py`：

```
python <本skill绝对路径>/scripts/validate_kernelbench_task.py \
  /abs/path/{op_name}.py --json
```

验证脚本同时执行静态检查（4 个组件齐全）和运行时检查（实例化、前向传播、NaN/Inf、一致性）。

**结果处理**：
- 输出 `[VALID]` + 来源是 Step 1 的用户原始文件 → 直接返回，任务结束
- 输出 `[VALID]` + 来源是 Step 3 新生成的文件 → 进入 Step 5
- 输出 `[INVALID]` → 根据错误信息修复代码，重新验证（最多 2 次）
- 重试 3 次仍失败 → 向用户报告错误，请求协助

### Step 5: 用户确认

**任务描述文件内容非用户提供的原始代码时，必须执行**

将完整的 `{op_name}.py` 内容展示给用户，使用 `question` 工具请求确认。
不通过则结合用户反馈返回 Step 3。

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 自包含 | 所有依赖函数必须内联，禁止 import 项目内模块 |
| 可执行 | `Model(*get_init_inputs()).forward(*get_inputs())` 必须直接运行 |
| 确定性 | 给定相同输入，输出必须一致 |
| 无 NaN/Inf | forward 输出不能包含 NaN 或 Inf |
| 禁止重写 | 原始函数可运行就直接复用，一行都不改 |
| 返回一致 | 返回类型/形状必须与原始实现一致 |
| 合理输入 | get_inputs 应提供合理大小的输入（不能过小或过大） |

---

## 输出格式

最终文件必须是**单一自包含 Python 文件**，包含以下 4 个部分：

```python
# 1. Imports 区（只允许标准库和 PyTorch 相关包）
import torch
import torch.nn as nn

# 2. Model 类
class Model(nn.Module):
    def __init__(self, <init_params>):
        super(Model, self).__init__()

    def forward(self, <forward_inputs>) -> torch.Tensor:
        return output

# 3. get_inputs()：返回 forward() 的输入参数列表
def get_inputs():
    input1 = torch.randn(batch_size, dim)
    input2 = torch.randn(batch_size, dim)
    return [input1, input2]

# 4. get_init_inputs()：返回 __init__() 的初始化参数列表
def get_init_inputs():
    return [dim_value]
```
