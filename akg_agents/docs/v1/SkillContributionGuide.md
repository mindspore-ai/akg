# Skill Documentation Contribution Guide

Welcome to contribute Skill documentation! This guide will help you quickly understand the Skill documentation specification.

---

## YAML Frontmatter Specification

### Required Fields

```yaml
---
name: triton-ascend-case-xxx               # Lowercase letters + hyphens
description: "Detailed description..."     # Include data scale, optimization techniques, performance metrics, applicable scenarios
level: L5                                  # L3 (basic/method), L4 (operator strategy), L5 (specific case)
category: example                          # fundamental, method, implementation, example
version: "1.0.0"                           # Version number
metadata:
  backend: ascend                          # Hardware backend
  dsl: triton-ascend                       # DSL type
  hardware: ascend910b4                    # Hardware model (optional)
---
```

### Field Descriptions

- **name**: Use lowercase letters and hyphens, following format `{dsl}-{type}-{specific-name}`
- **description**: **Core for LLM filtering**, must include:
  - Data scale (thousands/tens of thousands/millions of elements)
  - Core optimization techniques (e.g., compute reorganization, secondary splitting, atomic operations)
  - Performance metrics (optimal configuration, performance improvement ratio)
  - Applicable scenarios (specific data characteristics and problem types)
- **level**: 
  - L3: Basic concepts/general methods
  - L4: General strategies for a class of operators
  - L5: Specific cases with complete performance data
- **category**: fundamental/method/implementation/example (customizable)
- **metadata**: `backend`, `dsl`, `hardware` and other necessary metadata

---

## Content Structure

### 1. Task Characteristics (Required for L5 cases)
```markdown
## Task Characteristics
- **Operation Type**: elementwise, reduction, matmul, etc.
- **Data Size**: (M, N, K) and scale description
- **Data Types**: input/output data types
- **Task Features**: key characteristics and challenges
```

### 2. Optimization Points
```markdown
## Optimization: {Technique Name}

###  Simple Approach
\`\`\`python
# Unoptimized code
\`\`\`

### Optimized Approach
\`\`\`python
# Optimized code
\`\`\`

### Optimization Details
- Explain optimization principles
- Performance improvement data

### Summary
Key principles and applicable scenarios
```

### 3. Configuration Parameters (if applicable)
```markdown
## Autotune Configuration
\`\`\`python
configs = [
    triton.Config({'BLOCK_SIZE': 1024}),  # Performance: xxx
    triton.Config({'BLOCK_SIZE': 2048}),  # Performance: xxx   Optimal
]
\`\`\`
```

### 4. Summary
```markdown
### Summary
1. Key optimization principle 1
2. Key optimization principle 2
3. Applicable scenarios and boundary conditions
```

---

## Level Definition and Examples

| Level | Category | Description | Example |
|-------|----------|-------------|---------|
| L3 | fundamental | Core concepts, standard patterns | `triton-ascend-basics` |
| L3 | method | General optimization methods | `triton-ascend-optimization` |
| L4 | implementation | General strategies for operator classes | `triton-ascend-reduce` (sum/mean/max/softmax) |
| L5 | example | Specific cases + complete performance data | `triton-ascend-case-matmul-swizzle2d` |

---

## 📚 Reference Examples

We recommend referring to the following 4 documents to understand different types of Skill documentation:

### 1. Basic Documentation (L3/fundamental)
**File**: `python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-basics/SKILL.md`
- Shows basic concepts, core terminology, standard patterns

### 2. Operator Class Documentation (L4/implementation)
**File**: `python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-reduce/SKILL.md`
- Shows how to provide general optimization strategies for a class of operators (reduction)
- Covers multiple operators: sum/mean/max/softmax/layernorm

### 3. Specific Optimization Case (L5/example)
**File**: `python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-case-matmul-swizzle2d/SKILL.md`
- Shows how to write a complete specific optimization case
- Includes task characteristics, optimization techniques, code examples, performance data

### 4. General Optimization Documentation (L3/method)
**File**: `python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-optimization/SKILL.md`
- Shows general optimization techniques, API limitations, performance tuning methods

### More Examples
Browse the `python/akg_agents/op/resources/skills/triton-ascend/` directory for more Skill documentation examples:
- **Specific operator strategies**: `triton-ascend-elementwise`, `triton-ascend-matmul`, `triton-ascend-attention`
- **Optimization cases**: `triton-ascend-case-*` series (21 static shape optimization cases)
- **Tool documentation**: `triton-ascend-debugging`, `triton-ascend-memory`, `triton-ascend-grid-config`

---

## Key Principles

1. **Description is the core basis for LLM filtering**, must be information-rich
2. **Code examples use comparison format**, highlighting before/after differences
3. **Performance data must be authentic and reliable**, including specific configurations and test environments
4. **Avoid over-detailing metadata**, maintain generality and reusability
5. **Clear structure**, easy to understand and apply quickly

---

## Submission Process

1. Create Skill documentation following this specification
2. Place the file in the appropriate directory: `python/akg_agents/op/resources/skills/{dsl}/{skill-name}/SKILL.md`
3. Ensure documentation format is correct, code can run
4. Submit Pull Request with detailed description of optimization techniques and performance improvements

---

## Questions and Feedback

If you have questions about the Skill documentation specification, please:
- Refer to existing Skill documentation examples
- Submit an Issue on GitHub
- Contact the project maintainers

Thank you for your contribution!
