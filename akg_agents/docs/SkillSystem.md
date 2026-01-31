# Skill System

## Overview
Skill System is AIKG's knowledge management component for organizing, loading, and selecting professional knowledge (Skills). It follows Agent Skills open standard and supports LLM-driven intelligent selection. Primarily used for operator generation, also supports other domain extensions.

## Core Features
- **Unified Loading**: Load from standard location (`~/.akg/skills/`)
- **Intelligent Selection**: Two-stage filtering (metadata coarse + LLM fine)
- **Hierarchical Management**: L1-L5 levels (optional)
- **Version Management**: Multi-version with SemVer support
- **Installation Management**: Local and GitHub remote installation

---

## Core Components

### SkillRegistry - Skill Registry
```python
from ai_kernel_generator.core_v2.skill import SkillRegistry

registry = SkillRegistry()
count = registry.load_from_directory(Path("~/.akg/skills"))

# Query
skill = registry.get("cuda-basics")                           # Latest
skill = registry.get("cuda-basics", version="1.0.0")          # Specific
skill = registry.get("cuda-basics", strategy="oldest")        # Stable

# Query by level
l2_skills = registry.get_by_level(SkillLevel.L2)
```

### OperatorSkillSelector - Operator Selector
```python
from ai_kernel_generator.op.skill import (
    OperatorSkillSelector,
    OperatorSelectionContext
)

selector = OperatorSkillSelector()

context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

selected = selector.select(registry.get_all(), context, llm_func)
```

**Filters**: `backend_filter`, `dsl_filter`, `operator_type_filter`

### SkillInstaller - Installation Manager
```python
from ai_kernel_generator.core_v2.skill import SkillInstaller

installer = SkillInstaller()  # Default to ~/.akg/skills

# Local install
installer.install_from_directory(Path("./skills"))

# GitHub install
installer.install_from_github(repo="user/repo", skill_path="skills/my-skill")

# Uninstall
installer.uninstall("skill-name")
```

---

## SKILL.md Format

```markdown
---
name: cuda-basics
description: "CUDA programming basics"
level: L3
category: dsl
version: "1.0.0"
metadata:
  backend: "cuda"
  dsl: "cuda"
---

# CUDA Basics

Detailed Skill content (Markdown format)...
```

**Required**: `name` (lowercase letters, numbers, hyphens), `description`

**Optional**: `level` (L1-L5), `category`, `version`, `metadata`, `structure`

---

## Hierarchy (Optional)

| Level | Semantics | Examples |
|-------|-----------|----------|
| L1 | Process/Orchestration | standard-workflow |
| L2 | Component/Actor | coder-agent, verifier-agent |
| L3 | Method/Strategy | cuda-basics, triton-syntax |
| L4 | Implementation/Detail | error-handling |
| L5 | Atomic/Example | code-snippets |

---

## Typical Usage

### Operator Generation
```python
# 1. Load Skills
registry = SkillRegistry()
registry.load_from_directory(Path("~/.akg/skills"))

# 2. Create selector
selector = OperatorSkillSelector()

# 3. Define context
context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# 4. Select Skills
selected = selector.select(registry.get_all(), context, llm_func)

# 5. Build prompt
from ai_kernel_generator.core_v2.skill import build_prompt_with_skills
final_prompt = build_prompt_with_skills(selected, task_description)
```

### Other Domains
```python
from ai_kernel_generator.core_v2.skill import SkillSelector, SelectionContext

selector = SkillSelector()
context = SelectionContext(task_type="email_writing")
selected = selector.select(all_skills, context, llm_func)
```

---

## Version Management

```python
# Production: stable version
skill = registry.get("cuda-basics", strategy="oldest")

# Testing: latest version
skill = registry.get("cuda-basics", strategy="latest")

# Specific: exact version
skill = registry.get("cuda-basics", version="2.0.0")
```

---

## Directory Structure

**Code**: `python/ai_kernel_generator/core_v2/skill/`, `python/ai_kernel_generator/op/skill/`

**Skills**: `~/.akg/skills/` (unified installation location)

---

## Examples

Complete examples in `examples/run_skill/` directory:

| Example | Description |
|---------|-------------|
| `01_basic_usage.py` | Basic usage |
| `02_operator_generation.py` | Operator generation (core) |
| `03_skill_hierarchy.py` | Hierarchy management |
| `04_version_management.py` | Version management |
| `05_installer.py` | Installation mechanism |
| `06_url_install.py` | GitHub installation |
| `07_demo_email_writing.py` | Generalization demo |

See [`examples/run_skill/00_EXAMPLES_GUIDE.md`](../examples/run_skill/00_EXAMPLES_GUIDE.md)

---

**Last Updated**: 2026-01-30
