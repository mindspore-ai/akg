[English Version](../SkillContributionGuide.md)

# Skill 贡献指南

## 1. 概述

本指南说明如何为 AKG Agents 创建和贡献新的 Skill。Skill 是一个自包含的知识单元，Agent 可以动态加载以增强其能力。

## 2. SKILL.md 格式

每个 Skill 是一个包含 `SKILL.md` 文件的目录，文件包含 YAML frontmatter 和 Markdown 内容。

### 目录结构

```
my-skill/
├── SKILL.md          # 必需：Skill 定义
├── templates/        # 可选：Jinja2 模板
├── scripts/          # 可选：辅助脚本
└── examples/         # 可选：使用示例
```

### SKILL.md 模板

```markdown
---
name: my-skill-name
description: "清晰简洁地描述此 Skill 提供的知识"
category: guide
version: "1.0.0"
license: MIT
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Skill 标题

## 概述

领域知识的简要介绍。

## 核心概念

Markdown 格式的详细知识内容...

## 代码示例

```python
# 示例代码
```

## 最佳实践

- 实践 1
- 实践 2
```

## 3. YAML Frontmatter 字段

### 必需字段

| 字段 | 类型 | 规则 | 示例 |
|------|------|------|------|
| `name` | string | 小写字母数字 + 连字符，最长 64 字符 | `"cuda-basics"` |
| `description` | string | 1-1024 字符 | `"CUDA 编程基础知识"` |

### 推荐字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `category` | string | 语义类别 | `"guide"`、`"workflow"`、`"method"` |
| `version` | string | SemVer 版本号 | `"1.0.0"` |
| `license` | string | 许可证标识 | `"MIT"`、`"Apache-2.0"` |

### 可选字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `metadata` | dict | 自定义键值对，用于过滤 | `{backend: cuda, dsl: triton_cuda}` |
| `structure` | dict | 层级配置 | 详见 [Skill 系统](./SkillSystem.md) |

## 4. 命名规范

- 仅使用**小写**字母、数字和连字符
- 不使用下划线、空格或大写字母
- 描述性但简洁
- 示例：`ascend-memory-model`、`cuda-basics`、`triton-optimization`

## 5. 分类（category）指南

Skill 通过 `category` 字段进行语义分类，用于筛选与选择。标准分类包括：

| 分类 | 语义 | 说明 | 示例 |
|------|------|------|------|
| `workflow` | 流程/编排 | 高层工作流和编排知识 | `kernel-generation-workflow` |
| `overview` | 概览 | 系统或组件概览 | `system-overview` |
| `agent` | 组件/执行 | 组件级执行知识 | `kernel-designer-patterns` |
| `guide` | 设计与编程指南 | 设计方法论、编程指南 | `cuda-basics`、`triton-optimization` |
| `fundamental` | 基础 | 核心概念与原则 | `npu-architecture` |
| `method` | 方法/策略 | 优化方法与策略模式 | `tiling-strategies` |
| `implementation` | 实现/细节 | 实现细节与技巧 | `memory-coalescing` |
| `reference` | 参考 | 参考文档 | `triton-api-reference` |
| `example` | 示例 | 代码示例 | `relu-triton-example` |
| `case` | 案例 | 具体案例模式 | `softmax-optimization-case` |

## 6. 内容指南

1. **聚焦具体**：每个 Skill 专注一个主题
2. **包含示例**：代码示例对 LLM Agent 非常有价值
3. **结构清晰**：使用标题、列表和代码块
4. **注重实用**：侧重"如何做"而非纯理论
5. **合理版本**：使用 SemVer（主版本.次版本.修订版本）

## 7. 安装与测试

### 本地测试

```bash
# 将 Skill 放到项目 skills 目录
cp -r my-skill/ .akg/skills/

# 或使用 SkillInstaller 安装
python -c "
from akg_agents.core_v2.skill import SkillInstaller
installer = SkillInstaller()
installer.install_from_directory('path/to/my-skill')
"
```

### 验证加载

```python
from akg_agents.core_v2.skill import SkillRegistry

registry = SkillRegistry()
registry.load_from_directory('.akg/skills')

skill = registry.get('my-skill-name')
print(f"已加载: {skill.name} v{skill.version}")
print(f"描述: {skill.description}")
```

## 8. 提交

1. 按照上述格式创建 Skill 目录
2. 本地测试确保能正确加载
3. 提交 Pull Request，将 Skill 添加到相应的 skills 目录
