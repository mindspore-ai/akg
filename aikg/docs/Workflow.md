# Workflow System Design Document

## Overview

Workflow is the core configuration management system in AI Kernel Generator that defines and controls the execution flow of agents through YAML configuration files. It provides the rule foundation for Conductor's intelligent scheduling, supports flexible workflow customization, and achieves optimal execution paths for different scenarios.

## Core Philosophy

- **Configuration-Driven**: Declaratively define workflows through YAML files, reducing code coupling
- **Intelligent Decision Making**: Combine LLM analysis with rule constraints for adaptive agent scheduling
- **Modular Design**: Support flexible agent combinations to meet different task requirements
- **Extensibility**: Easy to add new agents and customize execution logic
- **Constraint Management**: Built-in retry limits and step control to ensure system stability

## Workflow Configuration Structure

### Configuration File Format

```yaml
# Agent information definition
agent_info:
  agent_name:
    possible_next_agent: [next_agent1, next_agent2]
    output_format:
      parser_name: parser_name
      parser_definition:
        output_fields:
          field_name:
            field_type: str
            mandatory: true
            field_description: "Field description"

# Starting agent
start_agent: agent_name

# List of agents requiring mandatory LLM analysis (when these agents enter conductor, conductor forces LLM analysis)
mandatory_analysis: [agent1, agent2]

# Limitation information
limitation_info:
  required: # Required limitation information
    max_step: 20  # Maximum execution steps
  optional: # Optional limitation information
    repeat_limits:
      single_agent:
        agent_name: 3  # Maximum consecutive repeats for single agent
      sequences:
        sequence_name:
          pattern: [agent1, agent2]  # Sequence pattern
          max_repeats: 3             # Maximum sequence repeats
```

### Core Configuration Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| agent_info | dict | Yes | Defines configuration information for all agents |
| start_agent | str | Yes | Specifies the starting agent for the workflow |
| mandatory_analysis | list | No | List of agents requiring mandatory LLM analysis |
| limitation_info | dict | Yes | Execution limitation configuration |

## Predefined Workflow Types

For reference, the actual workflow configuration files can be found in:
- [`python/ai_kernel_generator/config/default_workflow.yaml`](../python/ai_kernel_generator/config/default_workflow.yaml)
- [`python/ai_kernel_generator/config/conductor_connect_all_workflow.yaml`](../python/ai_kernel_generator/config/conductor_connect_all_workflow.yaml)
- [`python/ai_kernel_generator/config/coder_only_workflow.yaml`](../python/ai_kernel_generator/config/coder_only_workflow.yaml)
- [`python/ai_kernel_generator/config/verifier_only_workflow.yaml`](../python/ai_kernel_generator/config/verifier_only_workflow.yaml)

### 1. Default Workflow (default_workflow.yaml)
**Use Case**: Standard kernel generation process, suitable for most scenarios

**Flow**: `designer` → `coder` ←→ `verifier` → `finish`

**Features**:
- Linear process with each stage focusing on a single task
- Supports rollback to coder when verification fails
- Suitable for scenarios with clear algorithms requiring complete design documents

### 2. Fully Connected Workflow (conductor_connect_all_workflow.yaml)
**Use Case**: Complex kernels requiring multiple rounds of iterative optimization

**Flow**: Supports flexible transitions between all agents

**Features**:
- Maximum flexibility with arbitrary transitions between agents
- Designer and coder can self-repair
- Suitable for complex algorithms and multi-round optimization scenarios

### 3. Coder-Only Workflow (coder_only_workflow.yaml)
**Use Case**: Clear algorithm design, only code implementation needed

**Flow**: `coder` ←→ `verifier` → `finish`

**Features**:
- Skips design phase, directly generates code
- Coder can self-repair
- Suitable for standard kernels or scenarios with sufficient reference implementations

### 4. Verifier-Only Workflow (verifier_only_workflow.yaml)
**Use Case**: Code already exists, only verification needed

**Flow**: `verifier` → `finish`

**Features**:
- Minimalist process for verification only
- Suitable for code quality checks and performance testing

## Agent Configuration Details

### Agent Information Structure

```yaml
agent_name:
  possible_next_agent: [list]  # List of possible next agents
  output_format:               # Output format definition (optional)
    parser_name: str           # Parser name
    parser_definition:         # Parser definition
      output_fields:           # Output field definitions
        field_name:
          field_type: str      # Field type
          mandatory: bool      # Whether required
          field_description: str # Field description
```

### Special Agent Descriptions

#### Designer
- **Responsibility**: Generate algorithm pseudocode or implementation sketches
- **Output Format**: Requires parser to handle code fields
- **Typical Next Step**: coder (code implementation)

#### Coder
- **Responsibility**: Convert designs into concrete implementation code
- **Output Format**: Requires parser to handle code fields
- **Typical Next Step**: verifier (verification)

#### Verifier
- **Responsibility**: Verify code correctness and performance
- **Output Format**: Programmatic verification, no parser needed
- **Typical Next Step**: finish (completion) or other agents (repair)

## Limitation Mechanisms

### Execution Step Limits
```yaml
limitation_info:
  required:
    max_step: 20  # Prevent infinite loops
```

### Repeat Limits
```yaml
limitation_info:
  optional:
    repeat_limits:
      # Single agent consecutive repeat limits
      single_agent:
        designer: 2  # Designer max 2 consecutive executions
        coder: 2     # Coder max 2 consecutive executions
      
      # Sequence repeat limits
      sequences:
        coder_verifier:
          pattern: [coder, verifier]  # coder->verifier sequence
          max_repeats: 3              # This sequence max 3 repeats
```

## Decision Mechanisms

### Automatic Decision Conditions
1. **No Available Agents**: Directly finish
2. **Single Available Agent and Not Mandatory Analysis**: Direct execution
3. **Parsing Failed and Retryable**: Retry current agent

### LLM Intelligent Decision Conditions
1. **Multiple Available Agents**: Need intelligent selection
2. **Single Available Agent but in mandatory_analysis List**: Need analysis and suggestions

## Custom Workflow Guide

### 1. Creating New Workflows

```yaml
# custom_workflow.yaml
agent_info:
  # Define custom agents or modify existing agent flows
  custom_optimizer:
    possible_next_agent: [verifier, custom_optimizer]
    output_format:
      parser_name: optimizer_parser
      parser_definition:
        output_fields:
          optimized_code:
            field_type: str
            mandatory: true
            field_description: "Optimized code"

start_agent: custom_optimizer

mandatory_analysis: [custom_optimizer]

limitation_info:
  required:
    max_step: 15
```

### 2. Using Custom Workflows

```python
# Specify custom workflow when initializing Task
task = Task(
    op_name="custom_op",
    task_desc="...",
    workflow="custom_workflow"  # Use custom workflow
)
```

### 3. Extending Existing Workflows

```yaml
# Extend based on default_workflow
agent_info:
  designer:
    possible_next_agent: [coder]  # Add optimizer option
  coder:
    possible_next_agent: [verifier]
  verifier:
    possible_next_agent: [finish, coder, optimizer]
  optimizer:  # New optimizer agent
    possible_next_agent: [verifier, optimizer]

start_agent: designer
mandatory_analysis: [optimizer]  # Force analysis for optimizer
```

## Best Practices

### Workflow Design Principles

1. **Clear Objectives**: Choose appropriate workflow types based on specific use cases
2. **Minimize Complexity**: Prioritize simple linear processes
3. **Reasonable Limits**: Avoid infinite loops by setting appropriate retry counts
4. **Mandatory Analysis Configuration**: Configure mandatory_analysis for critical decision points

### Configuration Recommendations

1. **Development Phase**: Use fully connected workflows for flexible debugging
2. **Production Environment**: Use constrained workflows to ensure stability
3. **Performance Optimization**: Use coder-only workflows focusing on code quality
4. **Quality Checks**: Use verifier-only workflows to validate existing code

### Extension Guidelines

1. **New Agent Development**:
   - Define clear input/output formats
   - Implement corresponding parsers
   - Configure appropriate paths in workflows

2. **Workflow Optimization**:
   - Adjust limitation parameters based on actual usage
   - Analyze agent execution history to optimize decision paths
   - Use mandatory_analysis to control critical decision points

3. **Debugging Support**:
   - View complete execution traces through Trace
   - Analyze Conductor decision logs
   - Adjust workflow configurations to verify effects

Through the Workflow system, AI Kernel Generator achieves flexible, controllable, and extensible task execution flow management, providing optimal agent collaboration solutions for different scenarios.
