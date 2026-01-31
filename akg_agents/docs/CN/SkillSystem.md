# Skill System 技能管理系统

## 概述
Skill System 是 AIKG 的知识管理组件，负责组织、加载和选择专业知识（Skills）。遵循 Agent Skills 开放标准，支持 LLM 驱动的智能决策。主要用于算子生成场景，也支持其他领域扩展。

## 核心功能
- **统一加载**：从 `~/.akg/skills/` 标准位置加载
- **智能选择**：两阶段筛选（metadata 粗筛 + LLM 精筛）
- **层级管理**：L1-L5 分层组织（可选）
- **版本管理**：多版本共存，支持 SemVer
- **安装管理**：本地安装和 GitHub 远程安装

---

## 核心组件

### SkillRegistry - 技能注册表
```python
from ai_kernel_generator.core_v2.skill import SkillRegistry

registry = SkillRegistry()
count = registry.load_from_directory(Path("~/.akg/skills"))

# 查询
skill = registry.get("cuda-basics")                           # 最新版本
skill = registry.get("cuda-basics", version="1.0.0")          # 指定版本
skill = registry.get("cuda-basics", strategy="oldest")        # 稳定版本

# 按层级查询
l2_skills = registry.get_by_level(SkillLevel.L2)
```

### OperatorSkillSelector - 算子专用选择器
```python
from ai_kernel_generator.op.skill import (
    OperatorSkillSelector,
    OperatorSelectionContext
)

selector = OperatorSkillSelector()

# 定义算子生成上下文
context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# 选择 Skills（支持 LLM 精筛）
selected = selector.select(registry.get_all(), context, llm_func)
```

**过滤器**：
- `backend_filter`：按后端过滤（cuda/ascend/rocm）
- `dsl_filter`：按 DSL 过滤（triton/cuda/swft）
- `operator_type_filter`：按算子类型过滤

### SkillInstaller - 安装管理器
```python
from ai_kernel_generator.core_v2.skill import SkillInstaller

installer = SkillInstaller()  # 默认 ~/.akg/skills

# 本地安装
installer.install_from_directory(Path("./skills"))

# GitHub 安装
installer.install_from_github(
    repo="user/repo",
    skill_path="skills/my-skill"
)

# 卸载
installer.uninstall("skill-name")
```

---

## SKILL.md 格式

```markdown
---
name: cuda-basics
description: "CUDA 编程基础知识"
level: L3
category: dsl
version: "1.0.0"
metadata:
  backend: "cuda"
  dsl: "cuda"
---

# CUDA 基础

详细的 Skill 内容（Markdown 格式）...
```

**必需字段**：`name`（小写字母数字+连字符）、`description`

**可选字段**：`level`（L1-L5）、`category`、`version`、`metadata`、`structure`

---

## 层级体系（可选）

| 层级 | 语义 | 示例 |
|------|------|------|
| L1 | 流程/编排层 | standard-workflow |
| L2 | 组件/执行层 | coder-agent, verifier-agent |
| L3 | 方法/策略层 | cuda-basics, triton-syntax |
| L4 | 实现/细节层 | error-handling |
| L5 | 原子/样例层 | code-snippets |

---

## 典型使用流程

### 算子生成场景
```python
# 1. 加载 Skills
registry = SkillRegistry()
registry.load_from_directory(Path("~/.akg/skills"))

# 2. 创建选择器
selector = OperatorSkillSelector()

# 3. 定义上下文
context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# 4. 选择 Skills
selected = selector.select(registry.get_all(), context, llm_func)

# 5. 拼接 prompt
from ai_kernel_generator.core_v2.skill import build_prompt_with_skills
final_prompt = build_prompt_with_skills(selected, task_description)
```

### 其他领域
```python
from ai_kernel_generator.core_v2.skill import SkillSelector, SelectionContext

selector = SkillSelector()
context = SelectionContext(task_type="email_writing")
selected = selector.select(all_skills, context, llm_func)
```

---

## 版本管理

```python
# 生产环境：使用稳定版本
skill = registry.get("cuda-basics", strategy="oldest")

# 测试环境：使用最新版本
skill = registry.get("cuda-basics", strategy="latest")

# 特定需求：指定版本
skill = registry.get("cuda-basics", version="2.0.0")
```

---

## 目录结构

**代码位置**：
```
python/ai_kernel_generator/
├── core_v2/skill/              # 通用 Skill 管理
│   ├── metadata.py, loader.py, registry.py
│   ├── hierarchy.py, installer.py, version.py
│   └── skill_selector.py
└── op/skill/                   # 算子生成专用
    └── operator_selector.py
```

**Skills 位置**：`~/.akg/skills/`（统一安装位置）

---

## 示例

完整示例位于 `examples/run_skill/` 目录：

| 示例 | 说明 |
|------|------|
| `01_basic_usage.py` | 基础使用 |
| `02_operator_generation.py` | 算子生成（核心） |
| `03_skill_hierarchy.py` | 层级管理 |
| `04_version_management.py` | 版本管理 |
| `05_installer.py` | 安装机制 |
| `06_url_install.py` | GitHub 安装 |
| `07_demo_email_writing.py` | 泛化演示 |

详见 [`examples/run_skill/00_EXAMPLES_GUIDE.md`](../../examples/run_skill/00_EXAMPLES_GUIDE.md)

---

**更新时间**：2026-01-30
