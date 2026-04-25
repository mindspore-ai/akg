# core_v2/skill/ — Skill 开发规范

## 职责

Skill 的注册、加载、层级管理、安装、动态选择和版本控制。

## 关键文件

| 文件 | 说明 |
|------|------|
| `metadata.py` | `SkillMetadata`、`SkillStructure`、标准分类（`STANDARD_CATEGORIES`） |
| `registry.py` | `SkillRegistry` — Skill 注册表 |
| `loader.py` | `SkillLoader` — 从文件系统加载 Skill |
| `hierarchy.py` | `SkillHierarchy` — 层级关系管理、循环检测 |
| `installer.py` | `SkillInstaller` — 安装 Skill 到 `~/.akg/skills/` |
| `skill_selector.py` | `SkillSelector`、`SelectionContext` — 按上下文动态选择 Skill |
| `version.py` | `Version`、`VersionManager`、`VersionStrategy` |

## 开发约定

### Skill 元数据格式

Skill 以 `SKILL.md` 文件定义，使用 YAML Frontmatter：

```yaml
---
name: my-skill
description: 面向 LLM 筛选的描述（说明适用场景，使 LLM 能准确匹配）
version: 1.0.0
category: guide          # fundamental / reference / guide / example / case
dsl: triton_ascend       # 下划线形式
metadata:
  operator_type: reduce  # guide 类必填：elementwise / reduce / matmul / attention
  case_type: fix         # case 类：fix / improvement
  framework: torch       # example 类：torch / mindspore / all
---
# Skill 正文（Markdown）
```

### Skill 目录结构（triton-ascend 为例）

```
op/resources/skills/triton-ascend/
├── fundamentals/       # L0 基础知识（始终注入）
├── guides/             # L1 算子类型指南（LLM 按需选择）
├── examples/           # L1 代码示例（跟随 guide 的 operator_type）
├── cases/              # L2 优化/修复案例（阶段性注入）
├── evolved-fix/        # L2 自进化修复经验
└── evolved-improvement/ # L2 自进化优化经验
```

### 新增 Skill 的标准流程

1. 在 `op/resources/skills/{dsl-key}/` 对应子目录下创建 `<skill-name>/SKILL.md`
2. YAML Frontmatter 必填 `name`、`description`、`version`、`category`、`dsl`
3. `description` 应面向 LLM 筛选场景编写，描述适用的算子类型和使用时机
4. guide 类需设置 `metadata.operator_type`，case 类需设置 `metadata.case_type`

### Skill 安装位置

- 包内 Skill：`op/resources/skills/`
- 用户级 Skill：`~/.akg/skills/`
- 项目级 Skill：`.akg/skills/`
- 自进化 Skill：`~/.akg/evolved_skills/{dsl}/`（通过软链接部署到包内目录）

## 不做什么

- **不要**在此写具体的 Skill 内容——归 `op/resources/skills/` 或 `workspace/.opencode/skills/`
- **不要**绕过 `SkillRegistry` 手动加载 Skill
- **不要**在 `dsl` 字段使用连字符形式——统一为下划线（`triton_ascend`，非 `triton-ascend`）

## 参考

- `docs/v2/SkillSystem.md` — Skill 系统设计
- `docs/v2/SkillContributionGuide.md` — 编写规范
- `docs/v2/SkillEvolution.md` — 自进化机制
- `docs/v2/KernelGen.md` — KernelGen 分层选择机制
