---
name: akg-issue
description: >
  生成符合 AKG 项目规范的 Issue 描述文件（Bug Report / RFC / Task）。
  支持两种模式：用户描述生成、基于分支 diff 自动生成。
  生成的 .md 和 .json 文件写入 .tmp/issue/，经规范校验后供用户确认和提交。
argument-hint: >
  两种使用方式：
  1. 描述模式：/akg-issue <问题描述>
  2. Diff 模式：/akg-issue 从当前分支到 <目标分支> 建 issue
  示例：/akg-issue softmax算子在batch>32时报IndexError
  示例：/akg-issue 帮我基于当前分支和目标分支origin/br_agents的差异提一个issue
---

# AKG Issue 生成

<role>
你是一个 Issue 构建助手。你的任务是帮助用户构建符合 AKG 项目规范的 Issue 描述文件。
支持两种模式：从用户描述生成，或从分支 diff 自动生成。
</role>

## ⛔ 核心规则

1. 所有产物写入 `$AKG_AGENTS_DIR/.tmp/issue/`，**禁止写入其他位置**。
2. Issue 模板格式已内嵌在本文档 Step 3a 中，**无需读取外部文件**，直接使用下方模板。
3. 所有 Issue 标题必须带 `[AKG_AGENTS]` 前缀。
4. 生成完成后**必须执行规范校验**。
5. API 提交仅在用户选择后执行。
6. **禁止在 shell 中打印 Token 值**。检查 Token 是否存在只能用 `[ -z "$VAR" ]` 判断。
7. **owner 和 repo 必须从 `git remote -v` 的 URL 提取，禁止从用户输入的分支名推断。**

---

## 流程总览

```
Step 0  模式判定（自动）
Step 1  收集信息（自动 / 交互）
Step 2  检测远程信息（自动）
Step 3  生成 Issue 文件（自动）
Step 4  规范校验（自动）
Step 5  展示结果 + 用户决策（一次交互）
Step 6  执行 API 提交（仅当用户选择"提交"时，不再二次确认）
```

---

## Step 0: 模式判定

解析 $ARGUMENTS，判定使用哪种模式：

**Diff 模式**触发条件（满足任一）：
- 输入包含"分支""diff""对比""比较""差异""branch"等关键词
- 输入格式为"从 A 到 B"或"A → B"
- 用户明确要求基于代码变更/提交生成 issue

**描述模式**：不满足以上条件的所有情况。

---

## Step 1: 收集信息

### Diff 模式

**目标分支解析**：从用户输入中提取目标分支名。用户输入的分支名直接作为 `git diff` 的目标 ref 使用，例如：
- 用户说"origin/master" → `git diff origin/master...HEAD`
- 用户说"master" → `git diff master...HEAD`

⚠️ **分支名仅用于 git diff/log 命令。不要从分支名推断 owner、repo 或平台信息。**

```bash
git rev-parse --abbrev-ref HEAD
git diff <target_ref>...HEAD --stat
git log <target_ref>..HEAD --oneline --no-merges
git diff <target_ref>...HEAD
```

无差异 → 报错终止。

基于 diff 自动推断（**不需要向用户询问**）：

| 字段 | 推断逻辑 |
|------|---------|
| `issue_type` | 代码变更类型：bug fix 相关 → `bug`，新功能 → `task`，大规模设计/重构 → `rfc` |
| `title` | `[AKG_AGENTS] <概要描述>` |
| `labels` | `["kind/<issue_type>"]` |
| `body` | 按对应模板格式生成，描述本分支做了什么、要解决什么问题 |

### 描述模式

从 $ARGUMENTS 中提取信息：

| 信息 | 提取方式 |
|------|---------|
| `issue_type` | 用户明确指定 → 直接用。否则推断：报错/crash/失败 → `bug`，需求/设计/方案 → `rfc`，其他 → `task` |
| `title` | 从描述提炼一句话摘要，加 `[AKG_AGENTS]` 前缀 |
| `body` | 按模板格式组织用户描述 |

**如果用户未指定 issue_type 且无法推断**，使用 `question` 工具**一次性**询问：

> 请选择 Issue 类型：
> 1. Bug Report — 报告一个 bug
> 2. RFC — 提出新功能或增强方案
> 3. Task — 跟踪一项任务

**Bug 类型额外采集环境信息**（自动执行）：

```bash
cat $HOME_DIR/.akg/check_env.md 2>/dev/null
uname -a
python3 --version 2>/dev/null || python --version
```

**信息不足时，一次性列出所有缺失字段**（不要逐个询问）：

> 请补充以下信息：
> 1. [Bug] 当前行为：实际发生了什么？
> 2. [Bug] 期望行为：你期望发生什么？
> 3. [Bug] 复现步骤
> （已自动采集：设备=ascend, Python=3.9.7, OS=Linux...）

---

## Step 2: 检测远程信息

**无论哪种模式**，都从 git remote 检测平台和仓库信息（用于后续提交）：

```bash
git remote -v
```

- **远程名**优先级：tracking remote > `origin` > `origin_gitcode` > 第一个
- **平台**：从 remote URL 识别 `gitcode.com` / `gitee.com` / `github.com`
- **owner/repo**：从 remote URL 提取（去掉 `.git` 后缀）

⚠️ **再次强调：owner 和 repo 只能从 remote URL 提取，不能从用户输入的分支名提取。**

---

## Step 3: 生成 Issue 文件

### 3a. 生成 .md 文件

文件名：`issue_<slug>_<YYYYMMDD_HHmmss>.md`（slug 从 title 取前几个词，空格转 `-`）

**必须使用以下内嵌模板**：

**Bug Report**（`issue_type == "bug"`）：

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
- **GCC/Compiler version (if compiled from source)**: <如有>

## Describe the current behavior

<当前行为描述>

## Describe the expected behavior

<期望行为描述>

## Steps to reproduce the issue
1. <步骤1>
2. <步骤2>
3. <步骤3>

## Related log / screenshot

<日志或截图>

## Special notes for this issue

<补充说明>
```

**RFC**（`issue_type == "rfc"`）：

```markdown
---
name: RFC
about: Use this template for the new feature or enhancement
labels: kind/feature or kind/enhancement
---

## Background
<背景描述：问题现状>

## Introduction
<方案设计：解决方案概述>

## Trail
| No. | Task Description | Related Issue(URL) |
| --- | ---------------- | ------------------ |
| 1   |                  |                    |
```

**Task**（`issue_type == "task"`）：

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
| 1   |                  |          |
```

### 3b. 生成 .json 元数据文件

文件名：`issue_<slug>_<YYYYMMDD_HHmmss>.json`

```json
{
  "version": "1.0",
  "type": "issue",
  "issue_type": "<bug|rfc|task>",
  "platform": "<gitcode|gitee|github>",
  "repo": "<owner/repo>",
  "title": "[AKG_AGENTS] <标题>",
  "labels": ["kind/<bug|feature|task>"],
  "assignees": [],
  "generated_at": "<ISO 8601 时间戳>",
  "body_file": "<对应的 .md 文件名>",
  "validation": { "passed": null, "errors": [], "warnings": [] }
}
```

### 3c. 写入文件

```bash
mkdir -p $AKG_AGENTS_DIR/.tmp/issue
```

---

## Step 4: 规范校验

```bash
python $AKG_AGENTS_DIR/.opencode/skills/akg-issue/scripts/validate_issue.py $AKG_AGENTS_DIR/.tmp/issue/<json_file>
```

校验规则（脚本已实现）：

**通用**：

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| ISS-000 | title 必须以 `[AKG_AGENTS]` 开头 | error |
| ISS-001 | title 至少 10 字符 | error |
| ISS-002 | labels 至少包含一个 `kind/*` 标签 | error |

**Bug Report 专项**：BUG-001 ~ BUG-006（/device、软件环境、当前/期望行为、复现步骤、日志）

**RFC 专项**：RFC-001 ~ RFC-003（背景、方案设计、任务拆分）

**Task 专项**：TSK-001 ~ TSK-003（任务描述、任务目标、子任务）

---

## Step 5: 展示结果 + 用户决策

**一次性展示**所有信息，让用户做**一次**决策：

> **Issue 预览**
>
> | 字段 | 值 |
> |------|-----|
> | 标题 | `[AKG_AGENTS] <标题>` |
> | 类型 | `<bug/rfc/task>` |
> | 标签 | `kind/<type>` |
> | 仓库 | `<owner>/<repo>` (<平台名>) |
>
> **正文摘要**：<前 200 字>
>
> **校验结果**：<逐条列出>
>
> **文件**：
> - 描述文件：`.tmp/issue/issue_<slug>_<ts>.md`
> - 元数据文件：`.tmp/issue/issue_<slug>_<ts>.json`
>
> 请选择：
> 1. **提交** — 通过 API 提交到 <平台名>
> 2. **仅保存** — 只保留本地文件
> 3. **修改** — 请说明要改什么
> 4. **放弃**

**处理逻辑**：
- **提交** → 直接执行 Step 6（不再二次确认）
- **仅保存** → 完成
- **修改** → 根据反馈修改 → 重新校验 → 重新展示本步骤
- **放弃** → 完成

---

## Step 6: API 提交

### 6a. 检查 Token

**⛔ 禁止使用 `echo $TOKEN` 或 `echo "${TOKEN:?}"` 打印 Token 值。**

```bash
# GitCode
if [ -z "$GITCODE_TOKEN" ]; then echo "GITCODE_TOKEN 未设置"; exit 1; fi
echo "GITCODE_TOKEN 已设置"

# Gitee
if [ -z "$GITEE_TOKEN" ]; then echo "GITEE_TOKEN 未设置"; exit 1; fi
echo "GITEE_TOKEN 已设置"

# GitHub
gh auth status
```

Token 未设置时提示：

> **GitCode**：登录 GitCode → 右上角头像 → 个人设置 → 访问令牌 → 新建访问令牌（勾选 api 权限）
> `export GITCODE_TOKEN="你的token"`
> 文档：https://docs.gitcode.com/docs/help/home/user_center/security_management/user_pat
>
> **Gitee**：设置 → 私人令牌 → 生成新令牌（勾选 issues 权限）
> `export GITEE_TOKEN="你的token"`
>
> **GitHub**：`gh auth login`

### 6b. 执行提交

**重要**：title、body 等内容在 Step 3 生成时已存在于上下文中，**无需重新读取文件**，直接使用即可。

**owner 和 repo 必须使用 Step 2 从 `git remote -v` 提取的值。**

**GitCode**（`platform == "gitcode"`）：

使用完整路径 `/repos/<owner>/<repo>/issues`，JSON 格式提交。

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/repos/<owner>/<repo>/issues?access_token=$GITCODE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "<title>",
    "body": "<body>"
  }'
```

| 参数 | 必填 | 位置 | 说明 |
|------|------|------|------|
| `access_token` | ✅ | query | Token 环境变量 |
| `owner` | ✅ | URL path | 从 `git remote -v` 的 URL 提取的组织/用户名 |
| `repo` | ✅ | URL path | 从 `git remote -v` 的 URL 提取的仓库名 |
| `title` | ✅ | JSON body | Step 3 生成的标题 |
| `body` | ❌ | JSON body | Step 3 生成的正文 |

**Gitee**（`platform == "gitee"`）：

```bash
curl -s -X POST "https://gitee.com/api/v5/repos/<owner>/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "'"$GITEE_TOKEN"'",
    "repo": "<repo>",
    "title": "<title>",
    "body": "<body>"
  }'
```

**GitHub**（`platform == "github"`）：

```bash
gh issue create --title "<title>" --body-file $AKG_AGENTS_DIR/.tmp/issue/<file>.md \
  --label "<label1>" --label "<label2>"
```

### 6c. 处理响应

- **成功**（JSON 含 `html_url`）→ 展示 Issue 链接
- **失败** → 展示完整错误信息，常见排查：
  - `403` → URL 中 owner 不正确，检查 `git remote -v` 输出
  - `401` → Token 无效或过期
