[English Version](../SkillSystem.md)

# Skill 系统

## 1. 概述

Skill 系统提供了向 Agent 动态注入领域知识的机制。一个 **Skill** 是一个自包含的知识单元，以标准的 `SKILL.md` 文件（YAML frontmatter + Markdown 内容）描述。

核心能力：

- **标准格式**：SKILL.md，YAML frontmatter 元数据
- **多路径加载**：项目级和全局级 Skill 目录
- **分级管理**：L1-L5 层级，支持父子关系
- **LLM 驱动选择**：两阶段筛选（元数据过滤 + LLM 选择）
- **版本管理**：基于 SemVer 的多版本支持
- **统一安装**：所有 Skill 安装到 `~/.akg/skills/`

## 2. SKILL.md 格式

每个 Skill 由一个 `SKILL.md` 文件定义，包含 YAML frontmatter：

```markdown
---
name: cuda-basics
description: "CUDA 编程基础与优化模式"
level: L3
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# CUDA 基础

Markdown 格式的领域知识内容...
```

### 必需字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | Skill 名称（小写字母数字 + 连字符，最长 64 字符） |
| `description` | string | Skill 描述（1-1024 字符） |

### 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `level` | SkillLevel | 层级：L1-L5 |
| `version` | string | SemVer 版本号（默认 `"1.0.0"`） |
| `category` | string | 语义类型（如 `workflow`、`agent`、`strategy`） |
| `license` | string | 许可证（MIT、Apache-2.0 等） |
| `metadata` | dict | 自定义键值对元数据，用于过滤 |
| `structure` | dict | 层级配置（child_skills、default_children、exclusive_groups） |

## 3. Skill 层级（L1-L5）

> **注意**：Level 层级体系当前仅针对**算子生成场景**设计，其他场景的 Level 定义待设计。

| 层级 | 语义 | 说明 |
|------|------|------|
| L1 | 流程/编排层 | 高层工作流和编排知识 |
| L2 | 组件/执行层 | 组件级执行知识 |
| L3 | 方法/策略层 | 方法和策略模式 |
| L4 | 实现/细节层 | 实现细节和技巧 |
| L5 | 原子/样例层 | 原子级示例和代码片段 |

## 4. SkillLoader

`SkillLoader` 从多个目录扫描和加载 `SKILL.md` 文件。

### 支持的路径

**项目级**（相对于项目根目录）：
- `.akg/skills/`
- `.claude/skills/`
- `.opencode/skill/`

**全局级**（`enable_global=True` 时）：
- `~/.akg/skills/`
- `~/.claude/skills/`
- `~/.config/opencode/skill/`

### 用法

```python
from akg_agents.core_v2.skill import SkillLoader

loader = SkillLoader(base_dir=Path("./my_project"))
skills = loader.load_all()  # 从所有发现的目录加载
```

## 5. SkillRegistry

`SkillRegistry` 管理已加载的 Skill，支持按名称、层级和版本索引。

```python
from akg_agents.core_v2.skill import SkillRegistry

# 从目录加载
registry = SkillRegistry()
registry.load_from_directory(Path("./skills"))

# 查询 Skill
all_skills = registry.get_all()
skill = registry.get("cuda-basics")                    # 最新版本
skill = registry.get("cuda-basics", version="1.0.0")   # 指定版本
skill = registry.get("cuda-basics", strategy="stable")  # 最旧（稳定）版本

# 按层级过滤
l3_skills = registry.get_by_level(SkillLevel.L3)
```

### 版本选择策略

| 策略 | 说明 |
|------|------|
| `"latest"` | 最新版本（默认，适合开发环境） |
| `"oldest"` | 最旧版本（适合生产环境 / 稳定性） |
| `"stable"` | `"oldest"` 的别名 |

## 6. SkillHierarchy

`SkillHierarchy` 管理 Skill 之间的父子关系，并提供互斥检查。

```python
from akg_agents.core_v2.skill import SkillHierarchy

hierarchy = SkillHierarchy(registry)

# 查询关系
children = hierarchy.get_children("parent-skill")    # 返回 List[str]
parents = hierarchy.get_parents("child-skill")        # 返回 List[str]
descendants = hierarchy.get_descendants("parent-skill")  # 返回 Set[str]

# 检查互斥冲突
conflict = hierarchy.check_exclusive_conflict(skill_metadata, active_skills={"skill-a", "skill-b"})
# 返回冲突的 skill 名称（str）或 None

# 注册表变更后重建层级关系
hierarchy.rebuild()
```

### 验证工具

```python
from akg_agents.core_v2.skill.hierarchy import validate_all, detect_cycles

# 验证所有层级约束（级别顺序、渐进式展示）
is_valid, errors = validate_all(hierarchy)

# 检测循环依赖
cycles = detect_cycles(hierarchy)  # 返回 List[List[str]]
```

### 层级配置

在 `SKILL.md` frontmatter 中：

```yaml
structure:
  child_skills:
    - child-skill-a
    - child-skill-b
  default_children:
    - child-skill-a
  exclusive_groups:
    - [child-skill-a, child-skill-b]
```

## 7. SkillSelector

`SkillSelector` 实现两阶段 Skill 选择：

1. **粗筛**：基于元数据的快速过滤，使用自定义过滤函数
2. **精筛**：LLM 驱动的智能选择，基于任务上下文

```python
from akg_agents.core_v2.skill import SkillSelector, SelectionContext

# 创建选择上下文
context = SelectionContext(
    custom_fields={"backend": "cuda", "dsl": "triton_cuda"},
    include_levels=[SkillLevel.L3, SkillLevel.L4],  # 可选
    exclude_levels=[SkillLevel.L1],                   # 可选
)

# 创建带自定义过滤器的选择器
selector = SkillSelector(custom_filters=[
    create_metadata_matcher("backend"),
    create_metadata_matcher("dsl"),
])

# 仅粗筛（不使用 LLM）
candidates = selector.coarse_filter(all_skills, context)

# 完整两阶段选择（粗筛 + LLM 精筛）
selected = await selector.select(
    all_skills, context,
    llm_generate_func=llm_func,       # 可选：用于精筛的 LLM 函数
    prompt_template=None,              # 可选：自定义 LLM prompt 模板
    level=SkillLevel.L3,              # 可选：按层级过滤
)

# 获取父 Skill 的子 Skill
children = selector.get_children_skills(parent_skill, all_skills)
```

### SelectionContext

`SelectionContext` 是传递选择条件的数据类：

| 字段 | 类型 | 说明 |
|------|------|------|
| `include_levels` | `Optional[List[SkillLevel]]` | 仅包含这些层级的 Skill |
| `exclude_levels` | `Optional[List[SkillLevel]]` | 排除这些层级的 Skill |
| `custom_fields` | `Dict[str, Any]` | 自定义键值对，用于元数据匹配 |

### Prompt 组装

```python
from akg_agents.core_v2.skill.skill_selector import build_prompt_with_skills

# 将选中的 Skill 注入 prompt
prompt = build_prompt_with_skills(
    selected_skills,
    task_description="生成一个 relu 算子",
    include_full_content=True
)
```

### 过滤器工具

| 函数 | 说明 |
|------|------|
| `create_metadata_matcher(context_field, metadata_field, match_mode)` | 元数据匹配器工厂函数。支持 `"include"` 和 `"exclude"` 模式。 |
| `create_level_filter(match_mode)` | 层级过滤器工厂函数。 |
| `and_filters(*filters)` | 以 AND 逻辑组合过滤器。 |
| `or_filters(*filters)` | 以 OR 逻辑组合过滤器。 |

## 8. SkillInstaller

`SkillInstaller` 管理 Skill 安装到标准位置（`~/.akg/skills/`）。

```python
from akg_agents.core_v2.skill import SkillInstaller

installer = SkillInstaller()

# 安装单个 Skill
result = installer.install(Path("./my-skill"))

# 批量安装目录下的多个 Skill
results = installer.install_from_directory(Path("./my_skills"))

# 从 URL 安装（如 GitHub 仓库）
result = installer.install_from_url("https://github.com/user/skills-repo.git")

# 从 GitHub 简写安装
result = installer.install_from_github("user/skills-repo", skill_path="skills/my-skill")

# 验证安装完整性
is_valid, error = installer.verify("cuda-basics")

# 列出已安装的 Skill
installed = installer.list_installed()

# 检查是否已安装
if installer.is_installed("cuda-basics"):
    path = installer.get_skill_path("cuda-basics")

# 解析已安装 Skill 中的资源文件
resource = installer.resolve_resource("cuda-basics", "templates/kernel.j2")

# 卸载 Skill
installer.uninstall("cuda-basics")
```

### 安装目录结构

```
~/.akg/skills/
├── skill-name-1/
│   ├── SKILL.md
│   ├── templates/
│   ├── scripts/
│   └── .install.json       # 文件清单 + 哈希
├── skill-name-2/
│   └── ...
└── .registry.json           # 全局安装清单
```

## 9. VersionManager

`VersionManager` 提供基于 SemVer 的版本管理。

```python
from akg_agents.core_v2.skill import Version, VersionManager

# 解析版本
v = Version.parse("1.2.3-alpha.1")

# 比较版本
from akg_agents.core_v2.skill import compare_versions
result = compare_versions("1.2.0", "1.3.0")  # 返回 -1（小于）
```
