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
category: guide
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
| `category` | string | Semantic category | `"guide"`, `"workflow"`, `"method"` |
| `version` | string | SemVer version | `"1.0.0"` |
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

## 5. Category Guidelines

Skills are classified by `category` for filtering and selection. Standard categories include:

| Category | Semantic | Description | Example |
|----------|----------|-------------|---------|
| `workflow` | Process / Orchestration | High-level workflow and orchestration knowledge | `kernel-generation-workflow` |
| `overview` | Overview | System or component overview | `system-overview` |
| `agent` | Component / Actor | Component-level execution knowledge | `kernel-designer-patterns` |
| `guide` | Design & Programming Guide | Design methodology, programming guides | `cuda-basics`, `triton-optimization` |
| `fundamental` | Fundamental | Core concepts and principles | `npu-architecture` |
| `method` | Method / Strategy | Optimization methods and strategy patterns | `tiling-strategies` |
| `implementation` | Implementation / Detail | Implementation details and techniques | `memory-coalescing` |
| `reference` | Reference | Reference documentation | `triton-api-reference` |
| `example` | Atomic / Example | Code examples | `relu-triton-example` |
| `case` | Case Study | Concrete case patterns | `softmax-optimization-case` |

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
