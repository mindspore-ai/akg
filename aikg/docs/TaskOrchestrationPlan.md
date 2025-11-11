# Task Orchestration Plan Configuration

## Overview

The Task Orchestration Plan Configuration (Plan for short) declares the complete runtime scheme for a kernel generation task and is loaded by `Task` at runtime to drive agent collaboration and execution.

The plan mainly includes:
- `agent_model_config`: assign LLM presets to each agent (presets from `core/llm/llm_config.yaml`).
- `workflow_config_path`: points to the workflow YAML that defines execution flow (see [Workflow System Design Document](./Workflow.md)).
- `docs_dir`: reference documentation directories for agents (see [Doc-Driven Integration Guide](./DocDrivenIntegration.md)).
- `log_dir`: root directory for task logs.
- `profile_settings`: performance testing parameters (e.g., `run_times`, `warmup_times`).
- `verify_timeout`: verification timeout in seconds.

## Configuration File Structure

The plan uses YAML format and mainly contains the following items:

```yaml
# Agent model configuration
agent_model_config:
  designer: model_name
  coder: model_name
  conductor: model_name
  api_generator: model_name

# Log configuration
log_dir: "~/aikg_logs"

# Workflow configuration
workflow_config_path: "config/workflow_file.yaml"

# Documentation directory configuration
docs_dir:
  designer: "path/to/designer/docs"
  coder: "path/to/coder/docs"

# Performance analysis configuration
profile_settings:
  run_times: 50
  warmup_times: 5

# Verification configuration
verify_timeout: 300  # Verification timeout in seconds
```

## Main Configuration Items

### 1. Agent Model Configuration (agent_model_config)

`agent_model_config` assigns the LLM preset used by each agent. Values must come from `core/llm/llm_config.yaml` (e.g., `deepseek_r1_default`). Common keys: `designer`, `coder`, `conductor`, `api_generator` (extensible). Invalid names will be rejected during validation.

```yaml
# Example agent model configuration
agent_model_config:
  designer: deepseek_r1_default
  coder: deepseek_r1_default
```

### 2. Log Configuration (log_dir)

Specifies the storage directory for task execution logs:

- **Format**: String path
- **Support**: Absolute and relative paths (supports `~` for user home directory)
- **Default**: `"~/aikg_logs"`
- **Final path shape**: A subdirectory like `Task_{random_id}` will be created under this directory at runtime.

### 3. Workflow Configuration (workflow_config_path)

- Points to the workflow YAML for this task, configuring AIKG's task execution flow. Example: `"config/default_workflow.yaml"`.
- See [Workflow System Design Document](./Workflow.md).

### 4. Documentation Directory Configuration (docs_dir)

See the [Doc-Driven Integration Guide](./DocDrivenIntegration.md).

```yaml
# Example docs configuration
docs_dir:
  designer: "resources/docs/triton_ascend_docs"    # Designer docs (e.g., DSL syntax)
  coder: "resources/docs/triton_ascend_docs"       # Coder docs (e.g., DSL syntax)
```

### 5. Performance Analysis Configuration (profile_settings)

Configures execution parameters for performance testing:

```yaml
profile_settings:
  run_times: 50      # Number of performance test runs
  warmup_times: 5    # Number of warmup runs
```

### 6. Verification Configuration (verify_timeout)

Sets the timeout for code verification:

- **Unit**: Seconds
- **Default**: 300 seconds (5 minutes)
- **Purpose**: Prevents verification process from waiting indefinitely

## Preset Plans

### Triton Configuration (default_triton_cuda_config.yaml / default_triton_ascend_config.yaml)

**Use Case**: Triton kernel development

**Features**:
- Uses predefined design language
- Targets Triton code generation
- Supports Ascend NPU / CUDA GPU backend

**Configuration Examples**: 
- [`config/default_triton_cuda_config.yaml`](../python/ai_kernel_generator/config/default_triton_cuda_config.yaml) (for CUDA backend)
- [`config/default_triton_ascend_config.yaml`](../python/ai_kernel_generator/config/default_triton_ascend_config.yaml) (for Ascend backend)

### SWFT Configuration (default_swft_config.yaml)

**Use Case**: SWFT kernel development on Huawei Ascend NPUs

**Features**:
- Uses predefined design language
- Targets SWFT code generation
- Supports Ascend NPU backend

## Usage

```python
from ai_kernel_generator.config.config_validator import load_config

# Option 1: Load preset by DSL (e.g., triton)
config = load_config(dsl="triton_ascend", backend="ascend")  # or "triton_cuda" for CUDA backend

# Option 2: Load by explicit plan path
config = load_config(config_path="python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

task = Task(op_name="relu", task_desc="...", dsl="triton_ascend", config=config)  # or "triton_cuda" for CUDA
```

### 2. Custom Configuration Override

```python
# Customize based on default configuration
config = load_config(dsl="triton_ascend", backend="ascend")  # or "triton_cuda" for CUDA backend

# Override specific configuration items
config.update({
    "log_dir": "/custom/log/path",
    "agent_model_config": {
        "designer": "custom_model",
        "coder": "custom_model"
    }
})
```

### 3. Creating a New Plan File

```yaml
# custom_orchestration_plan.yaml
agent_model_config:
  designer: model_name
  coder: model_name
  conductor: model_name

log_dir: "~/custom_logs"

workflow_config_path: "config/custom_workflow.yaml"

docs_dir:
  designer: "resources/docs/custom_design_docs"
  coder: "resources/docs/custom_dsl_docs"

profile_settings:
  run_times: 100
  warmup_times: 10

verify_timeout: 600
```

## Configuration Extension Guide

### 1. Adding New Agent Configurations

```yaml
agent_model_config:
  designer: model_name
  coder: model_name
  conductor: model_name
  optimizer: model_name    # New optimizer agent
```

### 2. Extending Documentation Directory Configuration

```yaml
docs_dir:
  designer: "path/to/design/docs"
  coder: "path/to/code/docs"
  optimizer: "path/to/optimization/docs"  # New optimization docs
```
