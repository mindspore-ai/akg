# Custom Documentation Configuration Guide

## Overview

The CustomDocs feature allows users to configure custom reference documents for different Agents (such as Designer and Coder) in AIKG, thereby improving the quality and precision of code generation. By providing relevant technical documentation, API specifications, or algorithm descriptions to Agents, AI can better understand context and generate more accurate code.

## Configuration Method

The CustomDocs feature is implemented through the DSLConfig configuration system, primarily configured in the `docs_dir` section of configuration files:

### Basic Configuration Structure

```yaml
docs_dir:
  designer: "path/to/designer/docs"  # Designer reference document directory
  coder: "path/to/coder/docs"        # Coder reference document directory
```

## Required Document Types

Based on code analysis, the following documents are required for proper Agent operation:

### Designer Required Documents
- `basic_docs.md` - DSL basic documentation

### Coder Required Documents
- `basic_docs.md` - DSL basic documentation
- `api/api.md` - API interface documentation
- `suggestion_docs.md` - Expert suggestion documentation
- `examples/` directory - Contains example files starting with framework names (e.g., `mindspore_*.py`)

## Usage Method

1. **Prepare document directory structure**:
```
your_docs_dir/
├── basic_docs.md
├── api/
│   └── api.md  
├── suggestion_docs.md
└── examples/
    ├── mindspore_example.py
    ├── torch_example.py
    └── ...
```

2. **Specify paths in configuration file**:
```yaml
docs_dir:
  designer: "path/to/your_docs_dir"
  coder: "path/to/your_docs_dir"
```

3. **Apply configuration**:
```python
from ai_kernel_generator import load_config

config = load_config("your_custom_config")
task = Task(op_name="custom_op", task_desc="...", config=config)
```

Through the CustomDocs feature, users can customize the most suitable reference documents for different DSLs and usage scenarios, significantly improving AIKG's code generation effectiveness. 