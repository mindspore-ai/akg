# Default DSL Configuration Document

## Overview

DefaultDSLConfig is the default configuration management system for different DSLs (Domain Specific Languages) in AI Kernel Generator. It provides preset configuration templates for each DSL (such as Triton, SWFT, etc.), including model configurations, document paths, workflow settings, etc., to simplify the user configuration process.

## Configuration File Structure

Default DSL configuration files use YAML format and mainly contain the following configuration items:

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

Defines the LLM models used by each agent:

| Agent Name | Type | Description |
|------------|------|-------------|
| designer | str | Model used by the designer agent |
| coder | str | Model used by the coder agent |
| conductor | str | Model used by the conductor agent |
| api_generator | str | Model used by the API generator agent |

**Example Configuration**:
```yaml
agent_model_config:
  designer: deepseek_r1_default
  coder: deepseek_r1_default
  conductor: deepseek_r1_default
  api_generator: deepseek_r1_default
```

### 2. Log Configuration (log_dir)

Specifies the storage directory for task execution logs:

- **Format**: String path
- **Support**: Absolute and relative paths (supports `~` for user home directory)
- **Default**: `"~/aikg_logs"`

### 3. Workflow Configuration (workflow_config_path)

Specifies the default workflow configuration file used by this DSL:

- **Format**: Path relative to project root directory
- **Purpose**: Defines agent execution flow and limitations
- **Example**: `"config/default_workflow.yaml"`

### 4. Documentation Directory Configuration (docs_dir)

Specifies reference documentation directories for different agents:

```yaml
docs_dir:
  designer: "resources/docs/aul_docs"    # Designer docs (e.g., AUL specifications)
  coder: "resources/docs/triton_docs"    # Coder docs (e.g., DSL syntax)
```

**Purpose**:
- Designer: Provides algorithm design specification documents
- Coder: Provides target DSL syntax and API documentation

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

## Predefined DSL Configurations

### Triton Configuration (default_triton_config.yaml)

**Use Case**: Triton kernel development on NVIDIA GPUs

**Features**:
- Uses AUL / Sketch as design language
- Targets Triton code generation
- Supports Ascend NPU / CUDA GPU backend

**Configuration Example**: [`python/ai_kernel_generator/config/default_triton_config.yaml`](../python/ai_kernel_generator/config/default_triton_config.yaml)

### SWFT Configuration (default_swft_config.yaml)

**Use Case**: SWFT kernel development on Huawei Ascend NPUs

**Features**:
- Uses AUL as design language
- Targets SWFT code generation
- Supports Ascend NPU backend

## Configuration Usage Methods

### 1. Direct Use of Predefined Configurations

```python
from ai_kernel_generator import load_config

# Load Triton default configuration
config = load_config("default_triton_config")

# Create task
task = Task(
    op_name="relu",
    task_desc="...",
    dsl="triton",
    config=config
)
```

### 2. Custom Configuration Override

```python
# Customize based on default configuration
config = load_config("default_triton_config")

# Override specific configuration items
config.update({
    "log_dir": "/custom/log/path",
    "agent_model_config": {
        "designer": "custom_model",
        "coder": "custom_model"
    }
})
```

### 3. Creating New DSL Configurations

```yaml
# custom_dsl_config.yaml
agent_model_config:
  designer: your_model
  coder: your_model
  conductor: your_model

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

## Best Practices

### Configuration Management Principles

1. **Version Control**: Include configuration files in version control for change tracking
2. **Environment Separation**: Use different configurations for development, testing, and production environments
3. **Modular Design**: Separate common configurations from specific configurations
4. **Documentation Sync**: Update related documentation when configurations change

### Performance Optimization Recommendations

1. **Model Selection**: Choose appropriate LLM models based on task complexity
2. **Log Management**: Set reasonable log levels to avoid generating too many log files
3. **Timeout Settings**: Adjust verification timeout based on hardware performance
4. **Performance Testing**: Appropriately adjust profile_settings parameters to balance accuracy and efficiency

### Troubleshooting

1. **Path Issues**: Ensure all path configurations are correct and files exist
2. **Model Configuration**: Verify LLM model names and API configurations are correct
3. **Permission Issues**: Ensure log directories have write permissions
4. **Memory Management**: Monitor memory usage during performance testing

Through the DefaultDSLConfig system, AI Kernel Generator provides standardized configuration management for different DSLs, simplifying the user configuration process and improving system usability and maintainability.
