[中文版](./CN/SkillContributionGuide.md)

# Skill Contribution Guide

## 1. Overview

This guide explains how to create and contribute new Skills to AKG Agents. A Skill is a self-contained knowledge unit that agents can dynamically load to enhance their capabilities.

## 2. SKILL.md Format

Each Skill is a directory containing a `SKILL.md` file with YAML frontmatter and Markdown content.

### Directory Structure

```
my-skill/
├── SKILL.md          # Required: Skill definition
├── templates/        # Optional: Jinja2 templates
├── scripts/          # Optional: Helper scripts
└── examples/         # Optional: Usage examples
```

### SKILL.md Template

```markdown
---
name: my-skill-name
description: "A clear, concise description of what this skill provides"
level: L3
category: dsl
version: "1.0.0"
license: MIT
metadata:
  backend: cuda
  dsl: triton_cuda
---

# Skill Title

## Overview

Brief introduction to the domain knowledge.

## Key Concepts

Detailed knowledge content in Markdown format...

## Code Examples

```python
# Example code
```

## Best Practices

- Practice 1
- Practice 2
```

## 3. YAML Frontmatter Fields

### Required Fields

| Field | Type | Rules | Example |
|-------|------|-------|---------|
| `name` | string | Lowercase alphanumeric + hyphens, max 64 chars | `"cuda-basics"` |
| `description` | string | 1-1024 characters | `"CUDA programming fundamentals"` |

### Recommended Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `level` | string | Skill hierarchy level (L1-L5) | `"L3"` |
| `version` | string | SemVer version | `"1.0.0"` |
| `category` | string | Semantic category | `"dsl"`, `"workflow"`, `"strategy"` |
| `license` | string | License identifier | `"MIT"`, `"Apache-2.0"` |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `metadata` | dict | Custom key-value pairs for filtering | `{backend: cuda, dsl: triton_cuda}` |
| `structure` | dict | Hierarchy config | See [Skill System](./SkillSystem.md) |

## 4. Naming Conventions

- Use **lowercase** letters, numbers, and hyphens only
- No underscores, spaces, or uppercase letters
- Be descriptive but concise
- Examples: `ascend-memory-model`, `cuda-basics`, `triton-optimization`

## 5. Level Guidelines

> Note: The Level hierarchy is currently designed for the kernel generation scenario. Other scenarios' level definitions are pending.

| Level | When to Use | Example |
|-------|-------------|---------|
| L1 | High-level workflow / orchestration knowledge | `kernel-generation-workflow` |
| L2 | Component-level execution patterns | `kernel-designer-patterns` |
| L3 | Method / strategy knowledge | `cuda-basics`, `triton-optimization` |
| L4 | Implementation details | `tiling-strategies`, `memory-coalescing` |
| L5 | Atomic examples / code snippets | `relu-triton-example` |

## 6. Content Guidelines

1. **Be specific**: Focus on one topic per Skill
2. **Include examples**: Code examples are highly valuable for LLM agents
3. **Use clear structure**: Headers, lists, and code blocks
4. **Keep it actionable**: Focus on "how to" rather than theory
5. **Version appropriately**: Use SemVer (MAJOR.MINOR.PATCH)

## 7. Installation and Testing

### Local Testing

```bash
# Place your skill in the project skills directory
cp -r my-skill/ .akg/skills/

# Or install using SkillInstaller
python -c "
from akg_agents.core_v2.skill import SkillInstaller
installer = SkillInstaller()
installer.install_from_directory('path/to/my-skill')
"
```

### Verify Loading

```python
from akg_agents.core_v2.skill import SkillRegistry

registry = SkillRegistry()
registry.load_from_directory('.akg/skills')

skill = registry.get('my-skill-name')
print(f"Loaded: {skill.name} v{skill.version}")
print(f"Description: {skill.description}")
```

## 8. Submission

1. Create your Skill directory following the format above
2. Test locally to ensure it loads correctly
3. Submit a pull request with your Skill added to the appropriate skills directory
