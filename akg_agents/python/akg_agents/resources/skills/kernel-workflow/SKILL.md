---
name: kernel-workflow
description: AI Kernel 算子生成与优化工作流程。当用户需要生成、验证或优化 kernel 算子时使用此 Skill。支持 Triton、CUDA C、C++、TileLang等多种后端 DSL。
---

# Kernel 算子生成与优化工作流程

## 流程概述

1. **分析用户输入** → 只有要求描述还是有 task 代码？
2. **验证/生成 task 代码** → 详见 `references/torch_task_verify.md`
3. **选择生成/优化方式** → 详见 `references/tool-selection.md`
4. **执行生成/优化** → 调用子Agent（如用户有额外需求，传递 `user_requirements`）
5. **返回结果** → 使用 `finish`

**注意：`user_requirements` 是可选参数。**

## 输入类型判断

1. **只有需求描述**（无代码）→ 直接 `call_op_task_builder`
2. **task 代码**（包含 `class Model(nn.Module)`）→ 加载 `references/torch_task_verify.md` 进行验证
3. **Kernel 代码需优化**（包含 `@triton.jit` / `__global__` 等）→ 验证 task 代码后加载 `references/tool-selection.md`

## 基本流程示例

### 示例 1：用户只有需求描述

```
用户: 生成一个 ReLU 算子

Think: 用户没有提供代码，直接生成 task。
Action: call_op_task_builder(user_request="生成一个 ReLU 算子")
Observation: {status: "READY", task_code: "...", op_name: "relu"}

Think: task 已生成，加载 tool-selection 指导选择生成方式。
Action: read_file(file_path="resources/skills/kernel-workflow/references/tool-selection.md")
...
```

### 示例 2：用户提供 Torch task 代码

```
用户: 帮我生成这段代码的 kernel: class Model(nn.Module)...

Think: 用户提供了代码，加载验证指南。
Action: read_file(file_path="resources/skills/kernel-workflow/references/torch_task_verify.md")
Observation: <验证指南>

Think: 按指南验证代码格式。
Action: execute_script(script_path="resources/skills/kernel-workflow/scripts/check_torch_code.py", args="--stdin --json", stdin_input="<代码>")
...
```

### 示例 3：用户提供 kernel 代码要求优化

```
用户: 帮我优化这段 Triton kernel: @triton.jit def kernel(...): ...
      对应的 task 是: class Model(nn.Module)...

Think: 用户提供了 kernel 和 task，先验证 task。
Action: read_file(file_path="resources/skills/kernel-workflow/references/torch_task_verify.md")
...验证通过后...
Action: read_file(file_path="resources/skills/kernel-workflow/references/tool-selection.md")
...
```

## 参考文档

1. `references/torch_task_verify.md` - task 代码验证与补全流程
2. `references/tool-selection.md` - 子Agent 选择与二次确认流程

## Scripts

1. `scripts/check_torch_code.py` - 验证 task 代码格式（参数：`--stdin --json`）

## 禁止行为

1. 不验证 task 代码直接调用子Agent
2. 用户未指定生成方式时自动选择
3. task 已生成后再调用 `call_op_task_builder`
4. 调用子Agent 前不进行二次确认
