---
name: akg-review
description: >
  提交前代码自审工具。检查 rebase 冲突、代码规范（ruff/bandit）、危险函数、SPEC.md 合规性。
  生成审查报告写入 .tmp/review/。当用户输入 /akg-review 或要求代码审查、提交前检查时使用。
argument-hint: >
  可选：TARGET_BRANCH（目标分支，默认 origin_gitcode/br_agents）。
  示例：/akg-review  /akg-review origin_gitcode/master
---

# AKG Review - 提交前代码自审

<role>
你是代码自审助手。使用现有工具（ruff、git）+ 自定义脚本，
在提交前检查 rebase 冲突、代码规范、危险函数、SPEC.md 合规性。
</role>

## ⛔ 核心规则

1. 所有产物写入 `$AKG_AGENTS_DIR/.tmp/review/`。
2. **优先使用现成工具**：`ruff`（代码规范）、`git`（rebase 检查）。
3. **自定义检查**仅覆盖项目特定规则（危险函数、包名、SPEC.md）。
4. **不执行任何修改**，只检查和报告。

**依赖检查**（自动）：
- `ruff` - 如未安装，自动执行 `pip install ruff`
- `git` - 系统自带

---

## 流程

### Step 0: 依赖检查（自动安装）

```bash
# 检查并安装必需工具
for tool in ruff bandit; do
  if ! command -v $tool &> /dev/null; then
    echo "$tool 未安装，正在安装..."
    pip install $tool -q
  fi
done

# 可选工具（已有就用，没有就跳过）
MYPY_AVAILABLE=$(command -v mypy &> /dev/null && echo "yes" || echo "no")
VULTURE_AVAILABLE=$(command -v vulture &> /dev/null && echo "yes" || echo "no")
```

### Step 1: 确定目标分支 + 收集变更

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
TARGET_BRANCH=${ARGUMENTS:-origin_gitcode/br_agents}

# 收集变更的 Python 文件
CHANGED_FILES=$(git diff $TARGET_BRANCH...HEAD --name-only --diff-filter=ACMR | grep '\.py$' || echo "")
```

---

### Step 2: Rebase 冲突检查

```bash
python $AKG_AGENTS_DIR/.opencode/skills/akg-review/scripts/check_rebase.py \
  --target-branch "$TARGET_BRANCH" \
  --current-branch "$CURRENT_BRANCH" \
  --repo-path "$AKG_AGENTS_DIR" \
  --output /tmp/rebase_result.json
```

---

### Step 3: 代码规范检查（多工具组合）

#### 3a. Ruff 检查（快速、全面）

```bash
if [ -n "$CHANGED_FILES" ]; then
  ruff check $CHANGED_FILES \
    --select=E,F,W,C,N \
    --output-format=json \
    > /tmp/ruff_result.json 2>&1 || true
fi
```

**检测**：语法、导入、命名、复杂度

#### 3b. Bandit 安全检查（危险函数、SQL 注入等）

```bash
if [ -n "$CHANGED_FILES" ]; then
  bandit $CHANGED_FILES \
    --format json \
    --output /tmp/bandit_result.json \
    2>&1 || true
fi
```

**检测**：exec/eval、hardcoded passwords、SQL injection、shell injection

#### 3c. Mypy 类型检查（可选，如已安装）

```bash
if [ "$MYPY_AVAILABLE" = "yes" ] && [ -n "$CHANGED_FILES" ]; then
  mypy $CHANGED_FILES \
    --no-error-summary \
    --show-column-numbers \
    > /tmp/mypy_result.txt 2>&1 || true
fi
```

#### 3d. Vulture 死代码检测（可选，如已安装）

```bash
if [ "$VULTURE_AVAILABLE" = "yes" ] && [ -n "$CHANGED_FILES" ]; then
  vulture $CHANGED_FILES \
    --min-confidence 80 \
    > /tmp/vulture_result.txt 2>&1 || true
fi
```

#### 3e. 自定义规则检查（项目特定）

```bash
python $AKG_AGENTS_DIR/.opencode/skills/akg-review/scripts/check_code_style.py \
  --files "$CHANGED_FILES" \
  --repo-path "$AKG_AGENTS_DIR" \
  --output /tmp/custom_style_result.json
```

**自定义检查**（4 条项目特定规则）：

| 规则 | 说明 | 级别 |
|------|------|------|
| CODE-001 | License 头（Apache 2.0, 2025-2026） | error |
| CODE-002 | 从 `core/` 导入（应用 `core_v2/`） | error |
| CODE-004 | 参数值不规范（backend/dsl/arch） | error |
| CODE-013 | 错误包名（ai_kernel_generator） | error |
| CODE-010 | TODO/FIXME 注释 | info |

#### 3f. Mypy 类型检查（可选）

如果环境中有 mypy，自动运行类型检查。

#### 3g. Vulture 死代码检测（可选）

如果环境中有 vulture，检测未使用的函数/变量。

---

### Step 4: SPEC.md 合规性检查

```bash
python $AKG_AGENTS_DIR/.opencode/skills/akg-review/scripts/check_spec_compliance.py \
  --files "$CHANGED_FILES" \
  --base-branch "$TARGET_BRANCH" \
  --repo-path "$AKG_AGENTS_DIR" \
  --output /tmp/spec_result.json
```

**检查规则**（7 条架构约束）：

| 规则 | 说明 | 级别 |
|------|------|------|
| SPEC-001 | `core_v2/` 中有业务逻辑 | error |
| SPEC-002 | `op/` 中有框架代码 | error |
| SPEC-003 | `core_v2/tests/` 中有新测试 | error |
| SPEC-004 | 新目录缺 `__init__.py` | error |
| SPEC-005 | 新增模块未更新 SPEC.md | warning |
| SPEC-006 | 修改接口未更新 SPEC.md | warning |
| SPEC-007 | 测试缺 marker（op-st/bench） | warning |

---

### Step 5: 生成报告

```bash
mkdir -p $AKG_AGENTS_DIR/.tmp/review

python $AKG_AGENTS_DIR/.opencode/skills/akg-review/scripts/generate_report.py \
  --current-branch "$CURRENT_BRANCH" \
  --target-branch "$TARGET_BRANCH" \
  --rebase-result /tmp/rebase_result.json \
  --ruff-result /tmp/ruff_result.json \
  --custom-style-result /tmp/custom_style_result.json \
  --spec-result /tmp/spec_result.json \
  --output-dir "$AKG_AGENTS_DIR/.tmp/review" \
  --repo-path "$AKG_AGENTS_DIR"
```

**报告格式**：

```markdown
# AKG Code Review Report

**分支**: <current> → <target>
**状态**: ✅ PASS / ⚠️ WARNING / ❌ FAIL

## 检查概览

| 检查项 | 状态 | 错误 | 警告 |
|--------|------|------|------|
| Rebase 冲突 | ✅/❌ | N | 0 |
| Ruff 检查 | ✅/❌ | N | N |
| 自定义规则 | ✅/❌ | N | 0 |
| SPEC.md 合规 | ✅/❌ | N | N |

## 主要问题

1. [CODE-012] 使用危险函数 exec(): ...
2. [SPEC-001] 在 core_v2/ 中写业务逻辑: ...
...

## 修复建议

...
```

---

### Step 6: 展示结果

展示审查摘要 + **PR Checklist**（用于贴到 PR）：

> **代码审查完成**
>
> **分支**: `<current>` → `<target>`
> **状态**: ✅ PASS / ⚠️ WARNING / ❌ FAIL
>
> **检查结果**:
> - Rebase: ✅ 无冲突 / ❌ N 个冲突
> - Ruff: ✅ 通过 / ❌ N 个问题
> - Bandit: ✅ 通过 / ❌ N 个安全问题
> - 自定义: ✅ 通过 / ❌ N 个错误
> - SPEC: ✅ 通过 / ❌ N 个错误
>
> **文件**:
> - 详细报告: `.tmp/review/review_<branch>_<ts>.md`
> - **PR Checklist**: `.tmp/review/checklist_<branch>_<ts>.md` 👈 **复制此文件内容到 PR 描述**
>
> <如果有错误，列出前 3 个>

**PR Checklist 示例**（`.tmp/review/checklist_*.md`）：

```markdown
## Pre-submit Review Checklist

**Branch**: `feature/xxx` → `origin_gitcode/br_agents`
**Review Time**: 2026-04-02 16:30
**Status**: ✅ PASS

### Checks Performed

- [x] Rebase conflict check
- [x] Code style check (ruff + bandit + custom)
- [x] SPEC.md compliance check

### Summary

- **Errors**: 0
- **Warnings**: 2
- **Files Changed**: 5

**Result**: ✅ All checks passed. Safe to merge.

---
*Generated by `/akg-review`*
```

**使用方式**：
1. 运行 `/akg-review`
2. 复制 `.tmp/review/checklist_*.md` 的内容
3. 粘贴到 PR 描述的末尾

---

## 工具分工

| 工具 | 检查内容 | 安装 |
|------|---------|------|
| **ruff** | 语法、导入、命名、复杂度、行长度 | 自动 |
| **bandit** | 安全漏洞（exec/eval/SQL 注入/密码硬编码） | 自动 |
| **mypy** | 类型检查 | 可选 |
| **vulture** | 死代码检测 | 可选 |
| **自定义脚本** | License、包名、参数值、SPEC.md | 内置 |
| **git worktree** | Rebase 冲突 | 系统自带 |

---

## 自定义规则详解

### CODE-001: License 头

**检测**: Apache 2.0 License，年份 2025-2026 或 2026

---

### CODE-002: 从 core/ 导入

**检测**: `from akg_agents.core.` → 应改为 `core_v2.`

---

### CODE-013: 错误包名

**检测**: `ai_kernel_generator`, `kernel_generator`, `akg_agent`（少 s）

**示例**:

```python
# ❌ 错误
from ai_kernel_generator.core import Coder

# ✅ 正确
from akg_agents.core_v2.agents import AgentBase
```

---

### SPEC-001/002: 代码放置位置

- `core_v2/`: 只放框架代码（Agent 基类、Workflow、LLM）
- `op/`: 只放算子场景代码（KernelGen、验证器）
- `cli/`: 只放命令行入口

**检测**: 关键词匹配（`op_name`, `kernel code`, `verify kernel` 等）

---

## 注意事项

1. **ruff 优先**: 通用规范由 ruff 检查，速度快、准确度高。
2. **自定义精简**: 只检查项目特定规则，避免重复。
3. **独立运行**: 各阶段独立，一个失败不影响其他。
4. **只读操作**: 不修改代码，只生成报告。
