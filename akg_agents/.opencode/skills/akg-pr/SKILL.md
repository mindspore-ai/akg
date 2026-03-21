---
name: akg-pr
description: >
  基于当前分支与目标分支的 diff，自动生成符合 AKG 项目规范的 PR 描述文件。
  生成的 .md 和 .json 文件写入 .tmp/pr/，经规范校验后供用户确认和提交。
argument-hint: >
  可选：TARGET_BRANCH（目标分支，默认 master）。
  可选：REMOTE（远程名，默认自动检测）。
---

# AKG PR 生成

<role>
你是一个 PR 构建助手。你的任务是分析当前分支与目标分支之间的差异，
基于 AKG 项目的 PR 模板自动生成完整的 PR 描述文件，并进行规范校验。
</role>

## ⛔ 核心规则

1. 所有产物写入 `$AKG_AGENTS_DIR/.tmp/pr/`，**禁止写入其他位置**。
2. 必须基于 `.github/PULL_REQUEST_TEMPLATE.md` 的格式生成内容，不得自创格式。
3. 生成完成后**必须执行规范校验**，校验不通过必须提示用户修复。
4. 提交操作由用户自行完成或通过 API 完成，本 skill **不执行 git push 或 API 调用**。

---

## 流程总览

```
Step 1  确定分支与远程信息
Step 2  收集差异数据
Step 3  生成 PR 描述 (.md) 和元数据 (.json)
Step 4  规范校验
Step 5  展示结果，请求用户确认
```

---

## Step 1: 确定分支与远程信息

执行以下命令收集 git 上下文：

```bash
git rev-parse --abbrev-ref HEAD
git remote -v
git config --get branch.$(git rev-parse --abbrev-ref HEAD).remote
```

- **当前分支**：从 `HEAD` 获取
- **目标分支**：用户指定的 `TARGET_BRANCH`，默认 `master`
- **远程名**：用户指定的 `REMOTE`，或从当前分支的 tracking remote 获取，或按以下优先级选择：`origin` > `origin_gitcode` > 第一个可用 remote
- **平台判断**：从 remote URL 中识别 `gitcode.com` / `gitee.com` / `github.com`

🛑 如果当前分支就是目标分支（如 `master`），**调用 `question` 工具**提示用户：

> 当前分支是目标分支（master），无法生成 PR。请切换到功能分支后重试。

---

## Step 2: 收集差异数据

依次执行以下命令：

```bash
# 变更概览
git diff <target_branch>...HEAD --stat

# commit 历史
git log <target_branch>..HEAD --oneline --no-merges

# 完整 diff（用于 AI 分析，可能很长）
git diff <target_branch>...HEAD

# 检查是否有未提交的变更
git status --short
```

⚠️ 如果有未提交的变更，提醒用户：

> 检测到未提交的变更，这些变更不会包含在 PR 中。建议先 commit 后再生成 PR。

---

## Step 3: 生成 PR 文件

### 3a. 分析与推断

从 diff 和 commit 历史中推断以下信息：

| 字段 | 推断逻辑 |
|------|---------|
| `kind` | 从 commit message 关键词推断：`fix`/`bug` → `bug`，`feat`/`add`/`support` → `feature`，其他 → `task` |
| `title` | 从 commit 历史生成简洁标题，格式：`<kind>: <概要描述>` |
| `fixes` | 从 commit message 中提取 `#数字`、`Fixes #数字`、`Close #数字` 等模式 |
| `description` | AI 总结：做了什么、为什么做、怎么做的 |
| `reviewer_notes` | 高亮重点变更、潜在风险、需要特别关注的文件 |

### 3b. 生成 .md 文件

文件名格式：`pr_<branch>_<YYYYMMDD_HHmmss>.md`

内容**严格遵循** `.github/PULL_REQUEST_TEMPLATE.md` 格式：

```markdown
**What type of PR is this?**

/kind <bug|task|feature>

**What does this PR do / why do we need it**:

<AI 生成的描述>

**Which issue(s) this PR fixes**:

Fixes #<issue_number>（如有）

**Special notes for your reviewers**:

<AI 生成的审查要点>

---

### 变更概览

<git diff --stat 输出>

### 关键变更说明

<按模块/目录分组的变更说明>
```

### 3c. 生成 .json 元数据文件

文件名格式：`pr_<branch>_<YYYYMMDD_HHmmss>.json`

```json
{
  "version": "1.0",
  "type": "pr",
  "source_branch": "<当前分支>",
  "target_branch": "<目标分支>",
  "remote": "<远程名>",
  "platform": "<gitcode|gitee|github>",
  "repo": "<owner/repo>",
  "title": "<PR 标题>",
  "kind": "<bug|task|feature>",
  "fixes": ["#123"],
  "labels": ["kind/<kind>"],
  "reviewers": [],
  "generated_at": "<ISO 8601 时间戳>",
  "body_file": "<对应的 .md 文件名>",
  "validation": {
    "passed": null,
    "errors": [],
    "warnings": []
  }
}
```

### 3d. 写入文件

```bash
mkdir -p $AKG_AGENTS_DIR/.tmp/pr
```

将 .md 和 .json 写入 `$AKG_AGENTS_DIR/.tmp/pr/`。

---

## Step 4: 规范校验

对生成的 PR 文件执行以下校验规则：

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| PR-001 | `/kind` 必须存在且为 `bug`/`task`/`feature` 之一 | error |
| PR-002 | PR 描述不得为空，至少 20 字 | error |
| PR-003 | bug fix 类 PR 必须关联 issue（`Fixes #`） | error |
| PR-004 | commit 历史中不应有 merge commit | warning |
| PR-005 | 变更文件与 PR 描述的模块应匹配（无无关变更） | warning |
| PR-006 | title 建议符合 `<kind>: <描述>` 格式 | warning |

**校验输出格式**：

```
✅ PR-001: /kind 已设置 (feature)
✅ PR-002: 描述长度 156 字
❌ PR-003: bug fix PR 未关联 issue，请补充 Fixes #<issue_number>
⚠️  PR-004: 发现 2 个 merge commit，建议 rebase 清理
```

将校验结果回填到 .json 的 `validation` 字段。

- **存在 error** → 提示用户修复，不建议提交
- **仅 warning** → 提示但不阻止

---

## Step 5: 展示结果并请求确认

使用 `question` 工具向用户展示：

1. 生成的 PR 标题
2. PR 描述摘要
3. 校验结果
4. 文件存放路径

> PR 文件已生成：
> - 描述文件：`.tmp/pr/pr_<branch>_<ts>.md`
> - 元数据文件：`.tmp/pr/pr_<branch>_<ts>.json`
>
> 校验结果：<通过/未通过 + 详细信息>
>
> 请选择：
> 1. 确认，我将手动提交
> 2. 需要修改（请说明修改内容）
> 3. 放弃

**处理回复**：

- 确认 → 报告完成，提示用户提交方式（GitCode API / 手动操作）
- 修改 → 根据用户反馈修改文件，重新执行 Step 4
- 放弃 → 流程结束

---

## 提交指引（仅告知用户，skill 不执行）

生成完成后，可告知用户以下提交方式：

### GitCode API（需要配置 `GITCODE_TOKEN`）

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/<owner>/<repo>/pulls" \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "'$GITCODE_TOKEN'",
    "title": "<从 .json 读取>",
    "head": "<source_branch>",
    "base": "<target_branch>",
    "body": "<从 .md 读取>"
  }'
```

### Gitee API（需要配置 `GITEE_TOKEN`）

```bash
curl -X POST "https://gitee.com/api/v5/repos/<owner>/<repo>/pulls" \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "'$GITEE_TOKEN'",
    "title": "<从 .json 读取>",
    "head": "<source_branch>",
    "base": "<target_branch>",
    "body": "<从 .md 读取>"
  }'
```

### GitHub（需要 `gh` CLI 已登录）

```bash
gh pr create --title "<title>" --body-file .tmp/pr/<file>.md \
  --base <target_branch> --head <source_branch>
```
