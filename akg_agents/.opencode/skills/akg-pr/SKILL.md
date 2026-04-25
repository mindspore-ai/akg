---
name: akg-pr
description: >
  基于当前分支与目标分支的 diff，自动生成符合 AKG 项目规范的 PR 描述文件。
  生成的 .md 和 .json 文件写入 .tmp/pr/，经规范校验后供用户确认和提交。
argument-hint: >
  可选：TARGET_BRANCH（目标分支，默认 master）。
  可选：REMOTE（远程名，默认自动检测）。
  示例：/akg-pr  /akg-pr origin/br_agents
---

# AKG PR 生成

<role>
你是一个 PR 构建助手。你的任务是分析当前分支与目标分支之间的差异，
基于 AKG 项目的 PR 模板自动生成完整的 PR 描述文件，并进行规范校验。
</role>

## ⛔ 核心规则

1. 所有产物写入 `$AKG_AGENTS_DIR/.tmp/pr/`，**禁止写入其他位置**。
2. PR 模板格式已内嵌在本文档 Step 3b 中，**无需读取外部文件**，直接使用下方模板。
3. 所有 PR 标题必须带 `[AKG_AGENTS]` 前缀，格式：`[AKG_AGENTS] <kind>: <概要描述>`。
4. 生成完成后**必须执行规范校验**，校验不通过必须提示用户修复。
5. 本 skill **不执行 git push**。API 提交仅在用户选择后执行。
6. **禁止在 shell 中打印 Token 值**。检查 Token 是否存在只能用 `[ -z "$VAR" ]` 判断。

---

## 流程总览

```
Step 1  确定分支与远程信息（自动）
Step 2  收集差异数据（自动）
Step 3  生成 PR 描述文件（自动）
Step 4  规范校验（自动）
Step 5  展示结果 + 用户决策（一次交互）
Step 6  执行 API 提交（仅当用户选择"提交"时，不再二次确认）
```

---

## Step 1: 确定分支与远程信息

执行以下命令收集 git 上下文：

```bash
git rev-parse --abbrev-ref HEAD
git remote -v
git config --get branch.$(git rev-parse --abbrev-ref HEAD).remote
```

**参数解析规则**（解析 $ARGUMENTS）：

- 如果参数包含 `/`（如 `origin/master`）：
  - 先检查是否是已知 remote 名：执行 `git remote` 获取列表
  - `/` 前部分在 remote 列表中 → 该部分为 REMOTE，`/` 后为 TARGET_BRANCH
  - `/` 前部分**不在** remote 列表中 → 整体作为 TARGET_BRANCH 使用（可能是 `用户名/分支名` 格式的 ref）
- 无参数 → TARGET_BRANCH = `master`

**远程名**优先级：用户指定 > tracking remote > `origin` > `origin_gitcode` > 第一个

**平台与仓库**（⚠️ **必须从 `git remote -v` 的 URL 提取，禁止从用户输入的分支名推断**）：
- 从 remote URL 识别平台：`gitcode.com` / `gitee.com` / `github.com`
- 从 remote URL 提取 `owner/repo`（去掉 `.git` 后缀）

🛑 当前分支 == 目标分支 → 报错终止。

---

## Step 2: 收集差异数据

```bash
git diff <target_branch>...HEAD --stat
git log <target_branch>..HEAD --oneline --no-merges
git diff <target_branch>...HEAD
git status --short
```

⚠️ 无差异 → 报错终止。
⚠️ 有未提交变更 → 提醒但继续。

---

## Step 3: 生成 PR 文件

### 3a. 分析与推断

| 字段 | 推断逻辑 |
|------|---------|
| `kind` | 从 commit message 关键词推断：`fix`/`bug` → `bug`，`feat`/`add`/`support` → `feature`，其他 → `task` |
| `title` | `[AKG_AGENTS] <kind>: <概要描述>`。单 commit 时可直接用 message 加前缀 |
| `fixes` | 从 commit message 提取 `Fixes #N`、`Close #N`，**未检测到时写 N/A，不要编造**。关联 Issue 时必须使用**完整 URL**：`Fixes https://gitcode.com/mindspore/akg/issues/399` |
| `description` | AI 总结：做了什么、为什么做、怎么做的 |
| `reviewer_notes` | 重点变更、潜在风险、需要特别关注的文件 |

### 3b. 生成 .md 文件

文件名：`pr_<branch>_<YYYYMMDD_HHmmss>.md`

**必须使用以下模板格式**：

```markdown
**What type of PR is this?**

/kind <bug|task|feature>

**What does this PR do / why do we need it**:

<AI 生成的描述>

**Which issue(s) this PR fixes**:

Fixes #<issue_url>（如有，无则写 N/A，必须使用完整 URL 而不是数字编号）

**Special notes for your reviewers**:

<AI 生成的审查要点>

---

### 变更概览

<git diff --stat 输出>

### 关键变更说明

<按模块/目录分组的变更说明>
```

### 3c. 生成 .json 元数据文件

文件名：`pr_<branch>_<YYYYMMDD_HHmmss>.json`

```json
{
  "version": "1.0",
  "type": "pr",
  "source_branch": "<当前分支>",
  "target_branch": "<目标分支>",
  "remote": "<远程名>",
  "platform": "<gitcode|gitee|github>",
  "repo": "<owner/repo>",
  "title": "[AKG_AGENTS] <kind>: <概要描述>",
  "kind": "<bug|task|feature>",
  "fixes": ["#123"],
  "labels": ["kind/<kind>"],
  "reviewers": [],
  "generated_at": "<ISO 8601 时间戳>",
  "body_file": "<对应的 .md 文件名>",
  "validation": { "passed": null, "errors": [], "warnings": [] }
}
```

### 3d. 写入文件

```bash
mkdir -p $AKG_AGENTS_DIR/.tmp/pr
```

---

## Step 4: 规范校验

```bash
python $AKG_AGENTS_DIR/.opencode/skills/akg-pr/scripts/validate_pr.py $AKG_AGENTS_DIR/.tmp/pr/<json_file>
```

校验规则（脚本已实现）：

| 规则 ID | 描述 | 级别 |
|---------|------|------|
| PR-000 | title 必须以 `[AKG_AGENTS]` 开头 | error |
| PR-001 | `/kind` 必须存在且为 bug/task/feature | error |
| PR-002 | PR 描述至少 20 字 | error |
| PR-003 | bug fix PR 必须关联 issue（Fixes #） | error |
| PR-004 | 不应有 merge commit | warning |
| PR-006 | title 建议符合 conventional 格式 | warning |

校验结果回填到 .json 的 `validation` 字段。

---

## Step 5: 展示结果 + 用户决策

**一次性展示**所有信息，让用户做**一次**决策：

> **PR 预览**
>
> | 字段 | 值 |
> |------|-----|
> | 标题 | `[AKG_AGENTS] feature: 添加 xxx` |
> | 类型 | `/kind feature` |
> | 分支 | `<source_branch>` → `<remote>/<target_branch>` |
> | 仓库 | `<owner>/<repo>` (<平台名>) |
> | 关联 Issue | Fixes #123 / N/A |
>
> **校验结果**：<逐条列出>
>
> **文件**：
> - 描述文件：`.tmp/pr/pr_<branch>_<ts>.md`
> - 元数据文件：`.tmp/pr/pr_<branch>_<ts>.json`
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
> **Gitee**：设置 → 私人令牌 → 生成新令牌（勾选 pull_requests 权限）
> `export GITEE_TOKEN="你的token"`
>
> **GitHub**：`gh auth login`

### 6b. 执行提交

**重要**：title、body 等内容在 Step 3 生成时已存在于上下文中，**无需重新读取文件**，直接使用即可。

**owner 和 repo 必须使用 Step 1 从 `git remote -v` 提取的值，禁止从用户输入推断。**

**GitCode**（`platform == "gitcode"`）：

使用完整路径 `/repos/<owner>/<repo>/pulls`，JSON 格式提交。

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/repos/<owner>/<repo>/pulls?access_token=$GITCODE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "<title>",
    "head": "<source_branch>",
    "base": "<target_branch>",
    "body": "<body>"
  }'
```

| 参数 | 必填 | 位置 | 说明 |
|------|------|------|------|
| `access_token` | ✅ | query | Token 环境变量 |
| `owner` | ✅ | URL path | 从 `git remote -v` 的 URL 提取 |
| `repo` | ✅ | URL path | 从 `git remote -v` 的 URL 提取 |
| `title` | ✅ | JSON body | Step 3 生成的标题 |
| `head` | ✅ | JSON body | 当前分支名 |
| `base` | ✅ | JSON body | 目标分支名 |
| `body` | ❌ | JSON body | Step 3 生成的 PR 正文 |

**Gitee**（`platform == "gitee"`）：

```bash
curl -s -X POST "https://gitee.com/api/v5/repos/<owner>/<repo>/pulls" \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "'"$GITEE_TOKEN"'",
    "title": "<title>",
    "head": "<source_branch>",
    "base": "<target_branch>",
    "body": "<body>"
  }'
```

**GitHub**（`platform == "github"`）：

```bash
gh pr create --title "<title>" --body-file $AKG_AGENTS_DIR/.tmp/pr/<file>.md \
  --base <target_branch> --head <source_branch>
```

### 6c. 处理响应

- **成功**（JSON 含 `html_url`）→ 展示 PR 链接
- **失败** → 展示完整错误信息，常见排查：
  - `403` → URL 中 owner/repo 不正确，检查 `git remote -v`
  - `401` → Token 无效或过期
  - `422` → 分支未 push 或 PR 已存在
