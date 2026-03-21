---
name: akg-issue
description: >
  辅助构建符合 AKG 项目规范的 Issue 描述文件（Bug Report / RFC / Task）。
  生成的 .md 和 .json 文件写入 .tmp/issue/，经规范校验后供用户确认和提交。
argument-hint: >
  可选：ISSUE_TYPE（bug / rfc / task，不提供则交互式引导选择）。
  可选：用户对问题的初始描述文本。
---

# AKG Issue 生成

<role>
你是一个 Issue 构建助手。你的任务是通过交互式引导，帮助用户构建符合
AKG 项目规范的 Issue 描述文件，并进行规范校验。
</role>

## ⛔ 核心规则

1. 所有产物写入 `$AKG_AGENTS_DIR/.tmp/issue/`，**禁止写入其他位置**。
2. 必须基于 `.github/ISSUE_TEMPLATE/` 下对应模板的格式生成内容，不得自创格式。
3. 生成完成后**必须执行规范校验**，校验不通过必须提示用户修复。
4. 提交操作由用户自行完成或通过 API 完成，本 skill **不执行 API 调用**。

---

## 流程总览

```
Step 1  确定 Issue 类型
Step 2  收集信息（根据类型不同）
Step 3  生成 Issue 描述 (.md) 和元数据 (.json)
Step 4  规范校验
Step 5  展示结果，请求用户确认
```

---

## Step 1: 确定 Issue 类型

如果用户未指定 `ISSUE_TYPE`，使用 `question` 工具询问：

> 请选择 Issue 类型：
> 1. Bug Report — 报告一个 bug
> 2. RFC — 提出新功能或增强方案
> 3. Task — 跟踪一项任务

从用户的初始描述文本中也可推断类型（包含"报错""失败""崩溃"等关键词 → Bug；
包含"建议""需要""设计""方案"→ RFC；包含"任务""跟踪""计划"→ Task），
但推断结果仍需**确认**。

---

## Step 2: 收集信息

根据 Issue 类型，分别引导用户提供必要信息。**已有信息直接使用，缺失信息通过 `question` 工具逐项询问。**

### 2a. Bug Report（对应 `.github/ISSUE_TEMPLATE/bug-report.md`）

需要收集：

| 字段 | 必填 | 自动采集 | 说明 |
|------|------|---------|------|
| 硬件环境 | ✅ | ✅ 可从 `check_env.md` 获取 | Ascend/GPU/CPU |
| AKG 版本 | ✅ | ✅ 可从环境检测 | source or binary |
| Python 版本 | ✅ | ✅ `python --version` | — |
| OS 平台 | ✅ | ✅ `uname -a` | — |
| 当前行为 | ✅ | ❌ | 用户描述 |
| 期望行为 | ✅ | ❌ | 用户描述 |
| 复现步骤 | ✅ | ❌ | 用户描述 |
| 日志/截图 | ❌ | ❌ | 用户提供 |

**自动采集环境信息**：

先检查 `$HOME_DIR/.akg/check_env.md` 是否存在：
- 存在 → 从中提取硬件、Framework 版本等
- 不存在 → 执行以下命令采集：

```bash
uname -a
python3 --version 2>/dev/null || python --version
```

对于用户需要手动提供的字段，一次性询问（不要逐个问）：

> 请提供以下信息：
> 1. **当前行为**：实际发生了什么？
> 2. **期望行为**：你期望发生什么？
> 3. **复现步骤**：如何复现这个问题？（请尽量精确）
> 4. **相关日志/截图**（可选）：如有请粘贴

### 2b. RFC（对应 `.github/ISSUE_TEMPLATE/RFC.md`）

需要收集：

| 字段 | 必填 | 说明 |
|------|------|------|
| 背景 | ✅ | 问题现状描述 |
| 方案设计 | ✅ | 概要设计、伪代码 |
| 任务拆分 | ❌ | 子任务表格 |

> 请提供以下信息：
> 1. **背景**：要解决什么问题？当前现状是什么？
> 2. **方案设计**：你的解决方案概述（可包含伪代码）
> 3. **任务拆分**（可选）：需要拆分为哪些子任务？

### 2c. Task（对应 `.github/ISSUE_TEMPLATE/task-tracking.md`）

需要收集：

| 字段 | 必填 | 说明 |
|------|------|------|
| 任务描述 | ✅ | 任务内容 |
| 任务目标 | ✅ | 期望达成的目标 |
| 子任务 | ❌ | 子任务拆分 |

> 请提供以下信息：
> 1. **任务描述**：这个任务是什么？
> 2. **任务目标**：完成后达成什么效果？
> 3. **子任务拆分**（可选）：需要拆分为哪些子任务？

---

## Step 3: 生成 Issue 文件

### 3a. 生成 .md 文件

文件名格式：`issue_<slug>_<YYYYMMDD_HHmmss>.md`

`<slug>` 从 title 生成：取前 30 字符，中文保留，空格和特殊字符替换为 `-`，转小写。

内容**严格遵循**对应模板格式：

**Bug Report**：

```markdown
---
name: Bug Report
about: Use this template for reporting a bug
labels: kind/bug
---

## Environment
### Hardware Environment(`Ascend`/`GPU`/`CPU`):

/device <ascend|gpu|cpu>

### Software Environment:
- **AKG version (source or binary)**: <版本>
- **Python version**: <版本>
- **OS platform and distribution**: <系统信息>
- **GCC/Compiler version**: <如有>

## Describe the current behavior

<当前行为>

## Describe the expected behavior

<期望行为>

## Steps to reproduce the issue
1. <步骤1>
2. <步骤2>
3. <步骤3>

## Related log / screenshot

<日志或截图>

## Special notes for this issue

<补充说明>
```

**RFC**：

```markdown
---
name: RFC
about: Use this template for the new feature or enhancement
labels: kind/feature or kind/enhancement
---

## Background
<背景描述>

## Introduction
<方案设计>

## Trail
| No. | Task Description | Related Issue(URL) |
| --- | ---------------- | ------------------ |
| 1   | <任务1>          | <URL>              |
```

**Task**：

```markdown
---
name: Task
about: Use this template for task tracking
labels: kind/task
---

## Task Description

<任务描述>

## Task Goal

<任务目标>

## Sub Task
| No. | Task Description | Issue ID |
| --- | ---------------- | -------- |
| 1   | <子任务1>        | <ID>     |
```

### 3b. 生成 .json 元数据文件

文件名格式：`issue_<slug>_<YYYYMMDD_HHmmss>.json`

```json
{
  "version": "1.0",
  "type": "issue",
  "issue_type": "<bug|rfc|task>",
  "title": "<Issue 标题>",
  "labels": ["kind/<bug|feature|task>"],
  "assignees": [],
  "platform": "<gitcode|gitee|github>",
  "repo": "<owner/repo>",
  "generated_at": "<ISO 8601 时间戳>",
  "body_file": "<对应的 .md 文件名>",
  "validation": {
    "passed": null,
    "errors": [],
    "warnings": []
  }
}
```

**平台和仓库信息**：从 git remote 中检测，逻辑同 `akg-pr` skill。

### 3c. 写入文件

```bash
mkdir -p $AKG_AGENTS_DIR/.tmp/issue
```

将 .md 和 .json 写入 `$AKG_AGENTS_DIR/.tmp/issue/`。

---

## Step 4: 规范校验

对生成的 Issue 文件执行以下校验规则：

### 通用规则

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| ISS-001 | title 不得为空，至少 10 字符 | error |
| ISS-002 | labels 至少包含一个 `kind/*` 标签 | error |

### Bug Report 规则

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| BUG-001 | `/device` 必须存在且为 `ascend`/`gpu`/`cpu` 之一 | error |
| BUG-002 | 软件环境信息至少包含 AKG 版本和 Python 版本 | error |
| BUG-003 | "当前行为"不得为空 | error |
| BUG-004 | "期望行为"不得为空 | error |
| BUG-005 | "复现步骤"至少包含 1 个步骤 | error |
| BUG-006 | 建议附带相关日志 | warning |

### RFC 规则

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| RFC-001 | "背景"不得为空，至少 20 字符 | error |
| RFC-002 | "方案设计"不得为空，至少 30 字符 | error |
| RFC-003 | 建议包含任务拆分表格 | warning |

### Task 规则

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| TSK-001 | "任务描述"不得为空 | error |
| TSK-002 | "任务目标"不得为空 | error |
| TSK-003 | 建议包含子任务拆分 | warning |

**校验输出格式**：

```
✅ ISS-001: 标题长度 32 字符
✅ ISS-002: labels 包含 kind/bug
✅ BUG-001: /device 已设置 (ascend)
❌ BUG-005: 复现步骤为空，请补充
⚠️  BUG-006: 建议附带相关日志以便排查
```

将校验结果回填到 .json 的 `validation` 字段。

---

## Step 5: 展示结果并请求确认

使用 `question` 工具向用户展示：

1. 生成的 Issue 标题和类型
2. Issue 内容摘要
3. 校验结果
4. 文件存放路径

> Issue 文件已生成：
> - 描述文件：`.tmp/issue/issue_<slug>_<ts>.md`
> - 元数据文件：`.tmp/issue/issue_<slug>_<ts>.json`
>
> 校验结果：<通过/未通过 + 详细信息>
>
> 请选择：
> 1. 确认，我将手动提交
> 2. 需要修改（请说明修改内容）
> 3. 放弃

**处理回复**：

- 确认 → 报告完成，提示用户提交方式
- 修改 → 根据用户反馈修改文件，重新执行 Step 4
- 放弃 → 流程结束

---

## 提交指引（仅告知用户，skill 不执行）

### GitCode API（需要配置 `GITCODE_TOKEN`）

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/<owner>/<repo>/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "'$GITCODE_TOKEN'",
    "title": "<从 .json 读取>",
    "body": "<从 .md 读取>",
    "labels": "<从 .json 读取，逗号分隔>"
  }'
```

### Gitee API（需要配置 `GITEE_TOKEN`）

```bash
curl -X POST "https://gitee.com/api/v5/repos/<owner>/<repo>/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "'$GITEE_TOKEN'",
    "title": "<从 .json 读取>",
    "body": "<从 .md 读取>",
    "labels": "<从 .json 读取，逗号分隔>"
  }'
```

### GitHub（需要 `gh` CLI 已登录）

```bash
gh issue create --title "<title>" --body-file .tmp/issue/<file>.md \
  --label "<labels>"
```
