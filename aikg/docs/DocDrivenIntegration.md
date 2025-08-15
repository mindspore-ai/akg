# Doc-Driven Integration Guide (DDI)

## Overview

Doc-Driven Integration (DDI) treats documentation as a contract. By following a unified documentation specification (DocSpec) and plan declaration, new DSLs, frontends, and backends can be integrated without modifying the AIKG core, reducing coupling and maintenance costs.

## Orchestration Plan Integration

Enable and configure through the `docs_dir` field in the Task Orchestration Plan. Example: `aikg/python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml`:

### Basic Configuration Structure

```yaml
docs_dir:
  designer: "path/to/designer/docs"  # Designer reference document directory
  coder: "path/to/coder/docs"        # Coder reference document directory
```

## DocSpec Requirements

Based on code analysis, the following documents are required for proper Agent operation:

### Required
- `basic_docs.md` - DSL basic documentation

### Optional (Recommended)
- `basic_docs.md` - DSL basic documentation
- `api/api.md` - API interface documentation
- `suggestion_docs.md` - Expert suggestion documentation
- `examples/` directory - Contains example files starting with framework names (e.g., `mindspore_*.py`)

## Usage

1. Prepare the directory structure:
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

2. Specify paths in the plan file:
```yaml
docs_dir:
  designer: "path/to/your_docs_dir"
  coder: "path/to/your_docs_dir"
```

3. Apply the configuration:
```python
from ai_kernel_generator.config.config_validator import load_config

config = load_config(config_path="./your_custom_plan.yaml")
task = Task(op_name="custom_op", task_desc="...", config=config)
```

By following DocSpec and declaring `docs_dir` in the plan, you can efficiently integrate new DSLs/frontends/backends into AIKG.