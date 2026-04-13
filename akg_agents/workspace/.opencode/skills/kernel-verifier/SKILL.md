---
name: kernel-verifier
description: >
  算子代码验证 Skill — 静态代码检查 + 精度对比验证。
  包含两阶段验证：先做零成本静态检查（语法、编译、import、DSL 合规性），
  通过后再对比框架实现与生成实现的输出一致性。
  支持多框架（torch / mindspore）、多后端（cuda / ascend / cpu）。
argument-hint: >
  输入：op-name、dsl、backend、framework、verify-dir、code-file、命令模板。
  可选：device-id、timeout。
---

# Kernel Verifier Skill

<role>
你是一个内核代码验证专家。按照两阶段验证流程（静态检查 → 精度验证）确保生成代码的质量。
</role>

## 验证流程

Step 1 静态代码检查 
├─ 失败 → 结束
└─ 成功 → Step 2 创建验证项目 → Step 3 执行精度验证 → Step 4 收集结果

> 本 skill 加载后，`<base_url>` 标签提供 skill 目录路径（记为 **`$SD`**）。所有验证相关脚本路径基于 `$SD/scripts/`

### Step 1: 静态代码检查（快速预检）

在创建验证项目前，**先对生成代码进行静态检查**，快速拦截明显错误，避免浪费验证资源。

**使用** `bash` 工具调用本 skill 的 `scripts/code_check.py`（使用命令模板包裹）：

```bash
python3 $SD/scripts/code_check.py \
    --code_file <生成代码文件路径> \
    --backend <backend> \
    --dsl <dsl>
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--code_file` | 是 | 待检查的代码文件路径 |
| `--backend` | 否 | 后端（`cuda` / `ascend` / `cpu`） |
| `--dsl` | 否 | DSL（如 `triton_cuda`、`triton_ascend`，triton 系列会做 DSL 合规性检测） |

**检查内容**（纯静态分析，不调用 LLM，零额外成本）：

| 检查项 | 说明 |
|--------|------|
| 语法检查 | `ast.parse` 检测括号不匹配、缩进错误、关键字拼写等 |
| 编译检查 | `py_compile` 捕获额外编译问题 |
| import 可用性 | 检测代码中引用的模块是否在当前环境可用 |
| 中文文本混入 | 检测代码 token 中连续 ≥3 汉字（注释和字符串除外） |
| DSL 合规性 | 仅 triton 系列：检测是否定义了 `@triton.jit` kernel、是否通过 `kernel[grid](...)` 调用、forward() 是否使用了 torch 高层 API 替代 kernel |

**路由决策**：

| 结果 | 判断 | 动作 |
|------|------|------|
| 通过 | stdout 包含 `"静态检查通过"`，退出码 0 | 继续 Step 2 |
| 失败 | stdout 输出检查报告，退出码 1 | **不进入 Step 2**，将报告作为 `verifier_error` 反馈给代码生成步骤 |

### Step 2: 创建验证项目

在验证目录（如 `{output-path}/logs/iteration_{n}/verify/`）下创建两个文件：

- **`{op_name}_{framework}.py`**：复制任务文件完整内容（包含 `Model`, `get_inputs`, `get_init_inputs`）
- **`{op_name}_{dsl}_impl.py`**：复制生成代码完整内容（包含 `ModelNew`）

### Step 3: 执行精度验证

**必须使用** `bash` 工具调用本 skill 的 `scripts/verify.py`（使用命令模板包裹）：

```bash
python3 $SD/scripts/verify.py \
    --op_name <算子名> \
    --dsl <dsl> \
    --backend <backend> \
    --framework <framework> \
    --device_id <device_id> \
    --verify_dir <验证目录> \
    --timeout 300
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--op_name` | 是 | 算子名称 |
| `--dsl` | 是 | DSL（如 `triton_cuda`、`triton_ascend`） |
| `--backend` | 是 | 后端（`cuda` / `ascend` / `cpu`） |
| `--framework` | 否 | 框架（默认 `torch`） |
| `--device_id` | 否 | 设备 ID（默认 `0`，`-1` 自动选择） |
| `--verify_dir` | 否 | 验证目录（默认当前目录） |
| `--timeout` | 否 | 超时秒数（默认 300） |

⛔ 禁止自己编写测试代码替代此脚本。

### Step 4: 收集结果

| 结果 | 判断 |
|------|------|
| 验证通过 | stdout 包含 `"验证成功"`，退出码 0 |
| 验证失败 | stderr 包含错误信息，退出码非 0 |
| 超时 | stderr 包含 `"验证超时"`，退出码 1 |
