[中文版](./CN/SkillSystem.md)

# Skill System

## 1. Overview

The Skill System provides a mechanism for dynamically injecting domain knowledge into agents. A **Skill** is a self-contained knowledge unit described in a standard `SKILL.md` file (YAML frontmatter + Markdown content).

Key capabilities:

- **Standard format**: SKILL.md with YAML frontmatter metadata
- **Multi-path loading**: project-level and global-level skill directories
- **Category-based management**: category (e.g. workflow, agent, guide) with optional parent-child structure
- **LLM-driven selection**: two-phase filtering (metadata filter + LLM selection)
- **Version management**: SemVer-based multi-version support
- **Unified installation**: all skills installed to `~/.akg/skills/`

## 2. SKILL.md Format

Each Skill is defined by a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: cuda-basics
description: "CUDA programming fundamentals and optimization patterns"
category: guide
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# CUDA Basics

Markdown content with domain knowledge...
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Skill name (lowercase alphanumeric + hyphens, max 64 chars) |
| `description` | string | Skill description (1-1024 chars) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | Semantic type (e.g., `workflow`, `agent`, `guide`, `example`) |
| `version` | string | SemVer version (default: `"1.0.0"`) |
| `license` | string | License (MIT, Apache-2.0, etc.) |
| `metadata` | dict | Custom key-value metadata for filtering |
| `structure` | dict | Hierarchy config (child_skills, default_children, exclusive_groups) |

## 3. Skill Categories

Skills are classified by `category` for filtering and selection. Standard categories (see `STANDARD_CATEGORIES` in code) include:

| Category | Semantic | Description |
|----------|----------|-------------|
| workflow | Process / Orchestration | High-level workflow and orchestration knowledge |
| overview | Overview | System or component overview |
| agent | Component / Actor | Component-level execution knowledge |
| guide | Design & Programming Guide | Design methodology, programming guides |
| fundamental | Fundamental | Core concepts and principles |
| dsl | DSL Reference | Language or API reference |
| method | Method / Strategy | Optimization methods and strategy patterns |
| implementation | Implementation / Detail | Implementation details and techniques |
| reference | Reference | Reference documentation |
| example | Atomic / Example | Code examples |
| case | Case Study | Concrete case patterns |

## 4. SkillLoader

`SkillLoader` scans and loads `SKILL.md` files from multiple directories.

### Supported Paths

**Project-level** (relative to project root):
- `.akg/skills/`
- `.claude/skills/`
- `.opencode/skill/`

**Global-level** (when `enable_global=True`):
- `~/.akg/skills/`
- `~/.claude/skills/`
- `~/.config/opencode/skill/`

### Usage

```python
from akg_agents.core_v2.skill import SkillLoader

loader = SkillLoader(base_dir=Path("./my_project"))
skills = loader.load_all()  # Load from all discovered directories
```

## 5. SkillRegistry

`SkillRegistry` manages loaded skills with indexing by name, category, and version.

```python
from akg_agents.core_v2.skill import SkillRegistry

# Load from directory
registry = SkillRegistry()
registry.load_from_directory(Path("./skills"))

# Query skills
all_skills = registry.get_all()
skill = registry.get("cuda-basics")                    # Latest version
skill = registry.get("cuda-basics", version="1.0.0")   # Specific version
skill = registry.get("cuda-basics", strategy="stable")  # Oldest (stable) version

# Filter by category
guide_skills = registry.get_by_category("guide")
```

### Version Selection Strategies

| Strategy | Description |
|----------|-------------|
| `"latest"` | Most recent version (default, suitable for development) |
| `"oldest"` | Oldest version (suitable for production / stability) |
| `"stable"` | Alias for `"oldest"` |

## 6. SkillHierarchy

`SkillHierarchy` manages parent-child relationships between skills and provides mutual exclusion checking.

```python
from akg_agents.core_v2.skill import SkillHierarchy

hierarchy = SkillHierarchy(registry)

# Query relationships
children = hierarchy.get_children("parent-skill")    # Returns List[str]
parents = hierarchy.get_parents("child-skill")        # Returns List[str]
descendants = hierarchy.get_descendants("parent-skill")  # Returns Set[str]

# Check mutual exclusion conflict
conflict = hierarchy.check_exclusive_conflict(skill_metadata, active_skills={"skill-a", "skill-b"})
# Returns the conflicting skill name (str) or None

# Rebuild hierarchy after registry changes
hierarchy.rebuild()
```

### Validation Utilities

```python
from akg_agents.core_v2.skill.hierarchy import validate_all, detect_cycles

# Validate all hierarchy constraints (category order, progressive disclosure)
is_valid, errors = validate_all(hierarchy)

# Detect circular dependencies
cycles = detect_cycles(hierarchy)  # Returns List[List[str]]
```

### Hierarchy Configuration

In `SKILL.md` frontmatter:

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

`SkillSelector` implements two-phase skill selection:

1. **Coarse filtering**: metadata-based fast filtering using custom filter functions
2. **Fine selection**: LLM-driven intelligent selection based on task context

```python
from akg_agents.core_v2.skill import SkillSelector, SelectionContext

# Create selection context
context = SelectionContext(
    custom_fields={"backend": "cuda", "dsl": "triton_cuda"},
    include_categories=["guide", "implementation"],  # optional
    exclude_categories=["workflow"],                  # optional
    include_category_groups=["knowledge"],           # optional: e.g. orchestration, actor, knowledge, example
)

# Create selector with custom filters
selector = SkillSelector(custom_filters=[
    create_metadata_matcher("backend"),
    create_metadata_matcher("dsl"),
    create_category_filter("include"),
    create_category_filter("exclude"),
])

# Coarse filter only (no LLM)
candidates = selector.coarse_filter(all_skills, context)

# Full two-phase selection (coarse filter + LLM fine selection)
selected = await selector.select(
    all_skills, context,
    llm_generate_func=llm_func,       # optional: LLM function for fine selection
    prompt_template=None,              # optional: custom LLM prompt template
    category="guide",                  # optional: filter by category
)

# Get children skills of a parent
children = selector.get_children_skills(parent_skill, all_skills)
```

### SelectionContext

`SelectionContext` is a dataclass for passing selection criteria:

| Field | Type | Description |
|-------|------|-------------|
| `include_categories` | `Optional[List[str]]` | Only include skills with these categories |
| `exclude_categories` | `Optional[List[str]]` | Exclude skills with these categories |
| `include_category_groups` | `Optional[List[str]]` | Only include skills in these groups (orchestration, actor, knowledge, example) |
| `exclude_category_groups` | `Optional[List[str]]` | Exclude skills in these groups |
| `custom_fields` | `Dict[str, Any]` | Custom key-value pairs for metadata matching |

### Prompt Assembly

```python
from akg_agents.core_v2.skill.skill_selector import build_prompt_with_skills

# Build a prompt with selected skills injected
prompt = build_prompt_with_skills(
    selected_skills,
    task_description="Generate a relu kernel",
    include_full_content=True
)
```

### Filter Utilities

| Function | Description |
|----------|-------------|
| `create_metadata_matcher(context_field, metadata_field, match_mode)` | Factory for metadata matching filters. Supports `"include"` and `"exclude"` modes. |
| `create_category_filter(match_mode)` | Factory for category-based filters. |
| `and_filters(*filters)` | Combine filters with AND logic. |
| `or_filters(*filters)` | Combine filters with OR logic. |

## 8. SkillInstaller

`SkillInstaller` manages skill installation to the standard location (`~/.akg/skills/`).

```python
from akg_agents.core_v2.skill import SkillInstaller

installer = SkillInstaller()

# Install a single skill from a directory
result = installer.install(Path("./my-skill"))

# Batch install from a directory containing multiple skills
results = installer.install_from_directory(Path("./my_skills"))

# Install from URL (e.g., GitHub repository)
result = installer.install_from_url("https://github.com/user/skills-repo.git")

# Install from GitHub shorthand
result = installer.install_from_github("user/skills-repo", skill_path="skills/my-skill")

# Verify installation integrity
is_valid, error = installer.verify("cuda-basics")

# List installed skills
installed = installer.list_installed()

# Check if a skill is installed
if installer.is_installed("cuda-basics"):
    path = installer.get_skill_path("cuda-basics")

# Resolve a resource file within an installed skill
resource = installer.resolve_resource("cuda-basics", "templates/kernel.j2")

# Uninstall a skill
installer.uninstall("cuda-basics")
```

### Installation Directory Structure

```
~/.akg/skills/
├── skill-name-1/
│   ├── SKILL.md
│   ├── templates/
│   ├── scripts/
│   └── .install.json       # File manifest + hashes
├── skill-name-2/
│   └── ...
└── .registry.json           # Global installation manifest
```

## 9. VersionManager

`VersionManager` provides SemVer-based version management.

```python
from akg_agents.core_v2.skill import Version, VersionManager

# Parse version
v = Version.parse("1.2.3-alpha.1")

# Compare versions
from akg_agents.core_v2.skill import compare_versions
result = compare_versions("1.2.0", "1.3.0")  # Returns -1 (less than)
```
