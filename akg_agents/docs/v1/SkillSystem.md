# Skill System

## What is Skill System?

Skill System is the core knowledge management component of AKG Agents, enabling AI Agents to **load, select, and apply professional knowledge** to complete complex tasks.

### Core Philosophy

Imagine asking AI to generate a high-performance CUDA operator:
- **Traditional approach**: Dump all related documentation to AI and hope it finds useful parts
- **Skill System approach**: AI intelligently selects the most relevant knowledge pieces (like "Triton Syntax", "Reduce Patterns") based on task requirements (e.g., "generate softmax operator"), then completes the task

This is the value of Skill System: **Structuring vast knowledge so AI can precisely access needed information**.

### Key Features

- **Intelligent Selection**: Two-stage filtering (metadata coarse + LLM fine) to find the most relevant 3-5 Skills from hundreds
- **Unified Management**: All Skills installed in `~/.akg/skills/`, supporting local and GitHub remote installation
- **Version Control**: Multi-version coexistence, choose latest or stable versions
- **Extensible**: Domain-agnostic design, applicable to operator generation, documentation generation, test generation, etc.
- **Structured Knowledge**: Optional L1-L5 hierarchy system for organizing complex knowledge

---

## Quick Start

### Basic Usage (3 Steps)

```python
from pathlib import Path
from akg_agents.core_v2.skill import SkillRegistry

# 1. Create registry and load Skills
registry = SkillRegistry()
registry.load_from_directory(Path("~/.akg/skills"))

# 2. Query Skill
skill = registry.get("triton-basics")
print(f"Loaded Skill: {skill.name}")
print(f"Description: {skill.description}")

# 3. Use Skill content
print(skill.content)  # Knowledge content in Markdown format
```

---

## Core Concepts

### 1. Skill File Format (SKILL.md)

Each Skill is a file containing YAML metadata and Markdown content:

```markdown
---
name: triton-reduce
description: "Best practices for implementing Reduce operations in Triton"
level: L3
version: "1.0.0"
metadata:
  backend: "cuda"
  dsl: "triton"
  operator_patterns: "reduce, sum, max, min"
---

# Triton Reduce Patterns

## Overview
Reduce operations are core to many operators like softmax, layernorm...

## Basic Patterns
...detailed knowledge content...

## Example Code
...code examples...
```

**Metadata Fields**:
- **name** (required): Unique Skill identifier, lowercase letters/numbers/hyphens
- **description** (required): Brief description
- **level** (optional): Hierarchy marker (L1-L5)
- **version** (optional): Version number, default "1.0.0"
- **metadata** (optional): Custom tags for filtering (e.g., backend, dsl, operator_patterns)

### 2. Intelligent Selection Mechanism

Skill System uses two-stage selection to balance efficiency and accuracy:

**Stage 1: Coarse Filtering (Metadata)**
- Quickly filter out obviously irrelevant Skills
- Based on metadata tags (e.g., backend, dsl)
- Example: When generating CUDA operators, filter out all `backend: "ascend"` Skills

**Stage 2: Fine Selection (LLM Evaluation)**
- Let LLM read remaining candidates' descriptions and select the most relevant 3-5
- Consider task context and execution history
- Example: From 10 Triton-related Skills, select the 3 most suitable for softmax

```python
# Coarse filtering example
context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# Automatically filter out mismatched Skills
selector = OperatorSkillSelector()
candidates = selector.coarse_filter(all_skills, context)
# Result: 100 Skills → 10 candidates

# Fine selection example (LLM evaluation, requires prompt template)
prompt_template = """
Context: {context_str}
Candidate Skills: {skills_str}
Please select 1-3 most relevant Skills, return JSON format.
"""

selected = selector.select(
    all_skills, context, llm_func,
    prompt_template=prompt_template
)
# Result: 10 candidates → 3 most relevant Skills
```

### 3. Hierarchy System (Optional)

Skill System supports L1-L5 five-level hierarchy for organizing complex knowledge:

| Level | Semantics | Examples | Use Cases |
|-------|-----------|----------|-----------|
| **L1** | Process/Orchestration | `operator-generation-workflow` | Define overall process |
| **L2** | Component/Actor | `coder-agent`, `verifier-agent` | Describe Agent roles |
| **L3** | Method/Strategy | `triton-basics`, `reduce-pattern` | Specific technical knowledge |
| **L4** | Implementation/Detail | `error-handling`, `optimization-tips` | Details and techniques |
| **L5** | Atomic/Example | `code-templates`, `test-cases` | Code snippets and examples |

**Hierarchy Design Principles**:
- Upper-level Skills can reference lower-level Skills (via `structure.child_skills`)
- Supports progressive disclosure: load more details on demand
- Hierarchy is optional, simple scenarios can skip it

### 4. Version Management

Supports multiple versions of the same Skill:

```python
# Get latest version (default)
skill = registry.get("triton-basics")

# Get stable version (oldest)
skill = registry.get("triton-basics", strategy="oldest")

# Get specific version
skill = registry.get("triton-basics", version="1.2.0")

# View all versions
versions = registry.get_versions("triton-basics")
print(versions)  # ['1.0.0', '1.1.0', '1.2.0']
```

**Version Strategies**:
- **Development**: Use `latest` for newest features
- **Production**: Use `oldest` for stability
- **Specific needs**: Specify `version` to lock version

---

## Core Component APIs

### SkillRegistry - Registry

Manages loaded Skills, provides querying and version management:

```python
from akg_agents.core_v2.skill import SkillRegistry

registry = SkillRegistry()

# Load Skills
count = registry.load_from_directory(Path("~/.akg/skills"))
print(f"Loaded {count} Skills")

# Query single Skill
skill = registry.get("triton-basics")
skill = registry.get("triton-basics", version="1.0.0")
skill = registry.get("triton-basics", strategy="oldest")

# Query all Skills
all_skills = registry.get_all()

# Query by level
l3_skills = registry.get_by_level(SkillLevel.L3)

# Statistics
stats = registry.get_statistics()
print(stats)  # {'total': 50, 'by_level': {...}, ...}
```

### SkillSelector - Generic Selector

Domain-agnostic Skill selector supporting custom filters:

```python
from akg_agents.core_v2.skill import SkillSelector, SelectionContext

# Define context
context = SelectionContext(
    task_type="document_generation",
    custom_fields={
        "doc_type": "api",
        "language": "python"
    }
)

# Custom filter
def doc_type_filter(skill, context):
    doc_type = context.custom_fields.get("doc_type")
    if not doc_type:
        return True
    skill_types = skill.metadata.get("doc_types", "").split(",")
    return doc_type in skill_types

# Create selector
selector = SkillSelector(custom_filters=[doc_type_filter])

# Select Skills (must provide prompt template)
prompt_template = "Context: {context_str}\nCandidates: {skills_str}\nPlease select..."
selected = selector.select(all_skills, context, llm_func, prompt_template=prompt_template)
```

### OperatorSkillSelector - Operator-Specific Selector

Inherits `SkillSelector` with pre-configured operator generation filters:

```python
from akg_agents.op.skill import OperatorSkillSelector, OperatorSelectionContext

selector = OperatorSkillSelector()

context = OperatorSelectionContext(
    operator_type="softmax",
    dsl="triton",
    backend="cuda"
)

# Built-in filters:
# - backend_filter: filter by backend
# - dsl_filter: filter by dsl
# - operator_type_filter: filter by operator_patterns
# - hardware_filter: filter by hardware

# Select (must provide prompt template)
prompt_template = "Context: {context_str}\nCandidates: {skills_str}\nPlease select..."
selected = selector.select(all_skills, context, llm_func, prompt_template=prompt_template)
```

### SkillInstaller - Installation Manager

Manages Skill installation, updates, and uninstallation:

```python
from akg_agents.core_v2.skill import SkillInstaller

installer = SkillInstaller()  # Default install to ~/.akg/skills/

# Install single Skill locally
installer.install(Path("./my-skill"))

# Batch install all Skills in directory
installer.install_from_directory(Path("./skills"))

# GitHub installation
installer.install_from_github(
    repo="your-org/your-repo",
    skill_path="skills/triton-basics"
)

# Query installed Skills
installed = installer.list_installed()

# Uninstall Skill
installer.uninstall("triton-basics")

# Resolve Skill resource path (for Agent use)
template_path = installer.resolve_resource(
    skill_name="triton-basics",
    resource_path="templates/kernel.triton"
)
```

---

## Extending to New Domains

Skill System is domain-agnostic by design and can easily extend to any scenario.

### Universal Filter Tools

The framework provides universal filter tools, no need to write filtering logic manually:

**`create_metadata_matcher`**: Factory function to create metadata matchers

```python
from akg_agents.core_v2.skill import create_metadata_matcher

# Mode 1: Must include (default include mode)
backend_filter = create_metadata_matcher("backend")
# Meaning: context.backend value must appear in skill.metadata["backend"]

# Mode 2: Must exclude (exclude mode)
exclude_cpu = create_metadata_matcher("backend", "backend", "exclude")
# Meaning: context.backend value must NOT appear in skill.metadata["backend"]

# Mode 3: Different metadata field name
operator_filter = create_metadata_matcher("operator_type", "operator_patterns")
# Meaning: context.operator_type matches skill.metadata["operator_patterns"]
```

**Logic Combinators**:

```python
from akg_agents.core_v2.skill import and_filters, or_filters

# AND logic: all conditions must be satisfied
combined = and_filters(backend_filter, dsl_filter)

# OR logic: any condition must be satisfied
combined = or_filters(cuda_filter, ascend_filter)
```

### Example: Documentation Generation

```python
from dataclasses import dataclass
from akg_agents.core_v2.skill import (
    SelectionContext, 
    SkillSelector,
    create_metadata_matcher
)

# 1. Define domain-specific context
@dataclass
class DocSelectionContext(SelectionContext):
    doc_type: str = None      # api, tutorial, guide
    language: str = None      # python, java, cpp
    format: str = None        # markdown, rst, html

# 2. Use universal tools to create filters (recommended)
doc_type_filter = create_metadata_matcher("doc_type", "doc_types")
language_filter = create_metadata_matcher("language")

# Or write custom filter (for complex logic)
def custom_filter(skill, context):
    # Custom complex logic
    return True

# 3. Create selector
selector = SkillSelector(custom_filters=[
    doc_type_filter, 
    language_filter
])

# 4. Use (must provide prompt template)
context = DocSelectionContext(
    doc_type="api",
    language="python"
)

prompt_template = """
Context: {context_str}
Candidates: {skills_str}
Please select the most relevant documentation generation Skills.
"""

selected = selector.select(
    all_skills, context, llm_func,
    prompt_template=prompt_template
)
```

---

## Directory Structure

**Code Location**:
```
python/akg_agents/
├── core_v2/skill/              # Generic Skill management
│   ├── metadata.py             # Skill metadata definition
│   ├── loader.py               # Skill loader
│   ├── registry.py             # Skill registry
│   ├── skill_selector.py       # Generic selector
│   ├── installer.py            # Installation manager
│   ├── hierarchy.py            # Hierarchy management
│   └── version.py              # Version management
└── op/skill/                   # Operator-specific
    └── operator_selector.py    # Operator selector
```

**Example Code**: `examples/run_skill/`
```
examples/run_skill/
├── 01_basic_usage.py           # Basic usage example
├── 02_operator_generation.py  # Complete operator generation flow
├── 03_skill_hierarchy.py       # Hierarchy management example
├── 04_version_management.py   # Version management example
├── 05_installer.py             # Installation management example
├── 06_url_install.py           # GitHub remote installation example
├── 07_demo_email_writing.py   # Generalization demo
└── 08_prompt_assembly_triton_ascend.py  # Triton-Ascend Prompt assembly example
```

**Unit Tests**: `tests/v2/ut/test_skill_system.py`

**Skills Installation Location**: `~/.akg/skills/`

**Skills Directory Structure**:
```
~/.akg/skills/
├── triton-basics/
│   ├── SKILL.md
│   └── .install.json           # Installation info
├── reduce-pattern/
│   └── SKILL.md
└── .registry.json              # Global installation manifest
```

---

## FAQ

**Q: What's the difference between Skill System and RAG?**  
A: 
- RAG (Retrieval-Augmented Generation): Vector similarity-based retrieval, may retrieve many related but imprecise documents
- Skill System: Structured + two-stage filtering, precisely locates the most relevant knowledge pieces

**Q: Why is two-stage selection needed?**  
A: 
- Coarse filtering (metadata): Fast filtering, saves LLM cost
- Fine selection (LLM): Deep understanding, ensures selection accuracy

**Q: Is hierarchy system mandatory?**  
A: No. The hierarchy system (L1-L5) is optional, suitable for organizing complex knowledge. Simple scenarios can skip hierarchy.

**Q: How to handle Skill updates?**  
A: 
1. After modifying Skill file, increment version number
2. Run `installer.install()` for incremental update
3. Production can continue using old version, switch after testing
