# Task 代码格式验证指南

## 验证方式

使用 `execute_script` 执行验证脚本（同时进行静态检查和运行时检查）：

```python
execute_script(
    script_path="resources/skills/kernel-workflow/scripts/check_torch_code.py",
    args="--stdin --json",
    stdin_input="<task 代码>"
)
```

## KernelBench 格式要求

task 代码必须包含 4 个必需组件：

```python
import torch
import torch.nn as nn

class Model(nn.Module):         # 1. 必须继承 nn.Module
    def __init__(self):
        super().__init__()
        
    def forward(self, x):       # 2. Model 类的方法
        return torch.relu(x)

def get_inputs():               # 3. 顶层函数，返回 list
    return [torch.randn(16, 1024)]

def get_init_inputs():          # 4. 顶层函数，返回 list
    return []
```

## 使用场景

1. **只有需求描述** → 无需验证，直接 `call_op_task_builder`
2. **Torch task 代码** → 验证此代码，通过后加载 `tool-selection.md`
3. **Kernel 代码 + Task 代码** → 验证 task 代码，通过后加载 `tool-selection.md`

## 验证结果处理

1. `"valid": true` → 验证通过，加载 `tool-selection.md` 选择生成方式
2. `"valid": false` → 调用 `call_op_task_builder` 补全或修复：

```python
call_op_task_builder(user_request="补全以下代码为 KernelBench 格式：\n<原代码>")

call_op_task_builder(user_request="修复以下代码的错误：\n<原代码>\n\n错误信息：<error>")
```

---

## Task 相关需求的处理（重要！）

### 什么是 Task 需求

用户对 **task 代码**（而非 kernel 实现）的修改要求，这些需求通过 `call_op_task_builder` 处理，**不传递 `user_requirements`**：
例如：
1. 数据类型："输入改成 float16"、"使用 bfloat16"
2. 输入shape："batch_size 改成 64"、"dim 改为 2048"
3. 代码补全："补全 get_inputs"、"添加初始化"

### 处理示例

```
用户: "把输入改成 float16"
   → call_op_task_builder(user_request="修改 task 代码，将输入数据类型改为 float16：\n<原 task 代码>")

用户: "batch_size 改成 128，dim 改成 4096"
   → call_op_task_builder(user_request="修改 task 代码，batch_size=128，dim=4096：\n<原 task 代码>")
```
### 混合需求的处理

如果用户同时有 task 需求和 kernel 需求，分开处理：

```
用户: "生成 ReLU，输入 float16，核内二次切分"
         ↓
   1. task 需求 "float16" → call_op_task_builder(user_request="生成 float16 的 ReLU")
   2. 用户确认了task之后，kernel 需求 "核内二次切分" → call_coder_only(..., user_requirements="核内二次切分")
```
### 与 Kernel 需求的区分

- **Task 需求**：影响输入数据（类型、shape、 device设置与修改） → `call_op_task_builder`
- **Kernel 需求**：影响实现策略（切分、优化、算法） → `user_requirements` 参数

## 注意事项

1. **重要** 补全后无需重复验证（`call_op_task_builder` 内部保证格式正确）
2. 混合文本需先提取纯代码部分
3. **Task 相关需求不要传入 `user_requirements`**，应通过 `call_op_task_builder` 处理
