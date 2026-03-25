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
description: 一句话描述
version: 1.0.0
---
# Skill 正文（Markdown）
```

### 新增 Skill 的标准流程

1. 在 `op/resources/skills/<skill-name>/` 下创建 `SKILL.md`
2. YAML Frontmatter 填写 `name`、`description`、`version` 等元数据
3. 正文编写 Skill 指令内容
4. 如需附带脚本或参考文档，放在同目录的 `scripts/`、`references/` 子目录

### Skill 安装位置

- 包内 Skill：`op/resources/skills/`
- 用户级 Skill：`~/.akg/skills/`
- 项目级 Skill：`.akg/skills/`

## 不做什么

- **不要**在此写具体的 Skill 内容——归 `op/resources/skills/` 或 `workspace/.opencode/skills/`
- **不要**绕过 `SkillRegistry` 手动加载 Skill

## 参考

- `docs/v2/SkillSystem.md` — Skill 系统设计
- `docs/v2/SkillContributionGuide.md` — 编写规范
- `docs/v2/SkillEvolution.md` — 自进化机制
