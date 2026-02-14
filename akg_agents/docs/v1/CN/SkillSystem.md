# Skill System 技能管理系统

## 什么是 Skill System？

Skill System 是 AKG Agents 的核心知识管理组件，它让 AI Agent 能够**加载、选择和应用专业知识**来完成复杂任务。

### 核心理念

想象一下，你要让 AI 生成一个高性能的 CUDA 算子：
- **传统方式**：把所有相关文档都塞给 AI，希望它能找到有用的部分
- **Skill System 方式**：AI 根据任务需求（如"生成 softmax 算子"），智能选择最相关的知识片段（如"Triton 语法"、"Reduce 模式"），然后完成任务

这就是 Skill System 的价值：**将海量知识结构化，让 AI 能够精准获取所需信息**。

### 关键特性

- **智能选择**：两阶段筛选（metadata 粗筛 + LLM 精筛），从数百个 Skills 中找到最相关的 3-5 个
- **统一管理**：所有 Skills 安装在 `~/.akg/skills/`，支持本地和 GitHub 远程安装
- **版本控制**：支持多版本共存，可选择最新版或稳定版
- **可扩展**：设计上领域无关，可用于算子生成、文档生成、测试生成等任何场景
- **结构化知识**：可选的 L1-L5 层级体系，支持复杂知识的组织

---

## 快速开始

### 基础使用（3 步）

```python
from pathlib import Path
from akg_agents.core_v2.skill import SkillRegistry

# 1. 创建注册表并加载 Skills
registry = SkillRegistry()
registry.load_from_directory(Path("~/.akg/skills"))

# 2. 查询 Skill
skill = registry.get("triton-basics")
print(f"加载了 Skill: {skill.name}")
print(f"描述: {skill.description}")

# 3. 使用 Skill 内容
print(skill.content)  # Markdown 格式的知识内容
```

---

## 核心概念

### 1. Skill 文件格式（SKILL.md）

每个 Skill 是一个包含 YAML 元数据和 Markdown 内容的文件：

```markdown
---
name: triton-reduce
description: "Triton 中实现 Reduce 操作的最佳实践"
level: L3
version: "1.0.0"
metadata:
  backend: "cuda"
  dsl: "triton"
  operator_patterns: "reduce, sum, max, min"
---

# Triton Reduce 模式

## 概述
Reduce 操作是许多算子的核心，如 softmax、layernorm...

## 基本模式
...详细的知识内容...

## 示例代码
...代码示例...
```

**元数据说明**：
- **name**（必需）：Skill 唯一标识，小写字母数字+连字符
- **description**（必需）：简短描述
- **level**（可选）：层级标记（L1-L5）
- **version**（可选）：版本号，默认 "1.0.0"
- **metadata**（可选）：自定义标签，用于筛选（如 backend、dsl、operator_patterns）

### 2. 智能选择机制

Skill System 使用两阶段选择，平衡效率和准确性：

**阶段 1：粗筛（Metadata 过滤）**
- 快速过滤掉明显不相关的 Skills
- 基于 metadata 标签（如 backend、dsl）
- 示例：生成 CUDA 算子时，过滤掉所有 `backend: "ascend"` 的 Skills

**阶段 2：精筛（LLM 评估）**
- 让 LLM 阅读剩余候选的描述，选择最相关的 3-5 个
- 考虑任务上下文和执行历史
- 示例：从 10 个 Triton 相关 Skills 中，选出最适合 softmax 的 3 个

```python
# 粗筛示例
context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# 自动过滤掉不匹配的 Skills
selector = OperatorSkillSelector()
candidates = selector.coarse_filter(all_skills, context)
# 结果：100 个 Skills → 10 个候选

# 精筛示例（LLM 评估，需要提供 prompt 模板）
prompt_template = """
上下文: {context_str}
候选 Skills: {skills_str}
请选择最相关的 1-3 个 Skills，返回 JSON 格式。
"""

selected = selector.select(
    all_skills, context, llm_func,
    prompt_template=prompt_template
)
# 结果：10 个候选 → 3 个最相关的 Skills
```

### 3. 层级体系（可选）

Skill System 支持 L1-L5 五级分层，用于组织复杂知识：

| 层级 | 语义 | 示例 | 使用场景 |
|------|------|------|----------|
| **L1** | 流程/编排层 | `operator-generation-workflow` | 定义整体流程 |
| **L2** | 组件/执行层 | `coder-agent`, `verifier-agent` | 描述 Agent 角色 |
| **L3** | 方法/策略层 | `triton-basics`, `reduce-pattern` | 具体技术知识 |
| **L4** | 实现/细节层 | `error-handling`, `optimization-tips` | 细节和技巧 |
| **L5** | 原子/样例层 | `code-templates`, `test-cases` | 代码片段和示例 |

**层级设计原则**：
- 上层 Skill 可以引用下层 Skill（通过 `structure.child_skills`）
- 支持渐进式披露：按需加载更多细节
- 层级是可选的，简单场景可以不使用

### 4. 版本管理

支持同一 Skill 的多个版本共存：

```python
# 获取最新版本（默认）
skill = registry.get("triton-basics")

# 获取稳定版本（最旧版本）
skill = registry.get("triton-basics", strategy="oldest")

# 获取指定版本
skill = registry.get("triton-basics", version="1.2.0")

# 查看所有版本
versions = registry.get_versions("triton-basics")
print(versions)  # ['1.0.0', '1.1.0', '1.2.0']
```

**版本策略**：
- **开发环境**：使用 `latest` 获取最新特性
- **生产环境**：使用 `oldest` 保证稳定性
- **特定需求**：指定 `version` 锁定版本

---

## 核心组件 API

### SkillRegistry - 注册表

管理已加载的 Skills，提供查询和版本管理：

```python
from akg_agents.core_v2.skill import SkillRegistry

registry = SkillRegistry()

# 加载 Skills
count = registry.load_from_directory(Path("~/.akg/skills"))
print(f"加载了 {count} 个 Skills")

# 查询单个 Skill
skill = registry.get("triton-basics")
skill = registry.get("triton-basics", version="1.0.0")
skill = registry.get("triton-basics", strategy="oldest")

# 查询所有 Skills
all_skills = registry.get_all()

# 按层级查询
l3_skills = registry.get_by_level(SkillLevel.L3)

# 统计信息
stats = registry.get_statistics()
print(stats)  # {'total': 50, 'by_level': {...}, ...}
```

### SkillSelector - 通用选择器

领域无关的 Skill 选择器，支持自定义过滤器：

```python
from akg_agents.core_v2.skill import SkillSelector, SelectionContext

# 定义上下文
context = SelectionContext(
    task_type="document_generation",
    custom_fields={
        "doc_type": "api",
        "language": "python"
    }
)

# 自定义过滤器
def doc_type_filter(skill, context):
    doc_type = context.custom_fields.get("doc_type")
    if not doc_type:
        return True
    skill_types = skill.metadata.get("doc_types", "").split(",")
    return doc_type in skill_types

# 创建选择器
selector = SkillSelector(custom_filters=[doc_type_filter])

# 选择 Skills（需要提供 prompt 模板）
prompt_template = "上下文: {context_str}\n候选: {skills_str}\n请选择..."
selected = selector.select(all_skills, context, llm_func, prompt_template=prompt_template)
```

### OperatorSkillSelector - 算子专用选择器

继承 `SkillSelector`，预置算子生成过滤器：

```python
from akg_agents.op.skill import OperatorSkillSelector, OperatorSelectionContext

selector = OperatorSkillSelector()

context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# 内置过滤器：
# - backend_filter: 过滤 backend
# - dsl_filter: 过滤 dsl
# - operator_type_filter: 过滤 operator_patterns
# - hardware_filter: 过滤 hardware

# 选择（需要提供 prompt 模板）
prompt_template = "上下文: {context_str}\n候选: {skills_str}\n请选择..."
selected = selector.select(all_skills, context, llm_func, prompt_template=prompt_template)
```

### SkillInstaller - 安装管理器

管理 Skills 的安装、更新和卸载：

```python
from akg_agents.core_v2.skill import SkillInstaller

installer = SkillInstaller()  # 默认安装到 ~/.akg/skills/

# 本地安装单个 Skill
installer.install(Path("./my-skill"))

# 批量安装目录下所有 Skills
installer.install_from_directory(Path("./skills"))

# GitHub 安装
installer.install_from_github(
    repo="your-org/your-repo",
    skill_path="skills/triton-basics"
)

# 查询已安装的 Skills
installed = installer.list_installed()

# 卸载 Skill
installer.uninstall("triton-basics")

# 解析 Skill 资源路径（供 Agent 使用）
template_path = installer.resolve_resource(
    skill_name="triton-basics",
    resource_path="templates/kernel.triton"
)
```

---

## 扩展到新领域

Skill System 设计上是领域无关的，可以轻松扩展到任何场景。

### 通用过滤器工具

框架提供了通用的过滤器工具，无需手写过滤逻辑：

**`create_metadata_matcher`**：工厂函数，创建 metadata 匹配器

```python
from akg_agents.core_v2.skill import create_metadata_matcher

# 方式 1: 必须包含（默认 include 模式）
backend_filter = create_metadata_matcher("backend")
# 含义：context.backend 的值必须出现在 skill.metadata["backend"] 中

# 方式 2: 必须排除（exclude 模式）
exclude_cpu = create_metadata_matcher("backend", "backend", "exclude")
# 含义：context.backend 的值不能出现在 skill.metadata["backend"] 中

# 方式 3: metadata 字段名不同
operator_filter = create_metadata_matcher("operator_type", "operator_patterns")
# 含义：context.operator_type 匹配 skill.metadata["operator_patterns"]
```

**逻辑组合器**：

```python
from akg_agents.core_v2.skill import and_filters, or_filters

# AND 逻辑：所有条件都满足
combined = and_filters(backend_filter, dsl_filter)

# OR 逻辑：任一条件满足
combined = or_filters(cuda_filter, ascend_filter)
```

### 示例：文档生成

```python
from dataclasses import dataclass
from akg_agents.core_v2.skill import (
    SelectionContext, 
    SkillSelector,
    create_metadata_matcher
)

# 1. 定义领域特定上下文
@dataclass
class DocSelectionContext(SelectionContext):
    doc_type: str = None      # api, tutorial, guide
    language: str = None      # python, java, cpp
    format: str = None        # markdown, rst, html

# 2. 使用通用工具创建过滤器（推荐）
doc_type_filter = create_metadata_matcher("doc_type", "doc_types")
language_filter = create_metadata_matcher("language")

# 或手写过滤器（自定义逻辑）
def custom_filter(skill, context):
    # 自定义复杂逻辑
    return True

# 3. 创建选择器
selector = SkillSelector(custom_filters=[
    doc_type_filter, 
    language_filter
])

# 4. 使用（必须提供 prompt 模板）
context = DocSelectionContext(
    doc_type="api",
    language="python"
)

prompt_template = """
上下文: {context_str}
候选: {skills_str}
请选择最相关的文档生成 Skills。
"""

selected = selector.select(
    all_skills, context, llm_func,
    prompt_template=prompt_template
)
```

---

## 目录结构

**代码位置**：
```
python/akg_agents/
├── core_v2/skill/              # 通用 Skill 管理
│   ├── metadata.py             # Skill 元数据定义
│   ├── loader.py               # Skill 加载器
│   ├── registry.py             # Skill 注册表
│   ├── skill_selector.py       # 通用选择器
│   ├── installer.py            # 安装管理器
│   ├── hierarchy.py            # 层级管理
│   └── version.py              # 版本管理
└── op/skill/                   # 算子生成专用
    └── operator_selector.py    # 算子选择器
```

**示例代码**：`examples/run_skill/`
```
examples/run_skill/
├── 01_basic_usage.py           # 基础使用示例
├── 02_operator_generation.py  # 算子生成完整流程
├── 03_skill_hierarchy.py       # 层级管理示例
├── 04_version_management.py   # 版本管理示例
├── 05_installer.py             # 安装管理示例
├── 06_url_install.py           # GitHub 远程安装示例
├── 07_demo_email_writing.py   # 泛化应用示例
└── 08_prompt_assembly_triton_ascend.py  # Triton-Ascend Prompt 拼接示例
```

**单元测试**：`tests/v2/ut/test_skill_system.py`

**Skills 安装位置**：`~/.akg/skills/`

**Skills 目录结构**：
```
~/.akg/skills/
├── triton-basics/
│   ├── SKILL.md
│   └── .install.json           # 安装信息
├── reduce-pattern/
│   └── SKILL.md
└── .registry.json              # 全局安装清单
```

---

## 常见问题

**Q: Skill System 和 RAG 有什么区别？**  
A: 
- RAG（检索增强生成）：基于向量相似度检索，可能检索到大量相关但不精准的文档
- Skill System：结构化 + 两阶段筛选，精准定位最相关的知识片段

**Q: 为什么需要两阶段选择？**  
A: 
- 粗筛（metadata）：快速过滤，节省 LLM 成本
- 精筛（LLM）：深度理解，保证选择准确性

**Q: 必须使用层级体系吗？**  
A: 不是必须的。层级体系（L1-L5）是可选的，适合组织复杂知识。简单场景可以不使用层级。

**Q: 如何处理 Skill 更新？**  
A: 
1. 修改 Skill 文件后，增加版本号
2. 运行 `installer.install()` 进行增量更新
3. 生产环境可以继续使用旧版本，测试通过后再切换
