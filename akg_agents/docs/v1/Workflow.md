# Workflow System Design Document

## Overview
Workflow is AIKG's core configuration system that defines Agent execution flow through YAML configuration files, provides intelligent scheduling rules for Conductor, and supports flexible workflow customization.


## Configuration File Structure

### Basic Format
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

# Starting Agent
start_agent: agent_name

# List of agents requiring mandatory LLM analysis (when these agents enter conductor, conductor forces LLM analysis)
mandatory_llm_analysis: [agent1, agent2]

# Execution limitations
limitation_info:
  required: # Required limitation information
    max_step: 20  # Maximum execution steps
  optional: # Optional limitation information
    repeat_limits:
      single_agent:
        agent_name: 3                 # Maximum consecutive repeats for single Agent
      sequences:
        sequence_name:
          pattern: [agent1, agent2]   # Sequence pattern
          max_repeats: 3              # Maximum sequence repeats
```

### Core Configuration Fields
| Field Name | Type | Required | Description |
|---------|------|------|------|
| agent_info | dict | Yes | Define configuration information for all Agents |
| start_agent | str | Yes | Specify starting Agent |
| mandatory_llm_analysis | list | No | List of Agents requiring mandatory LLM analysis |
| limitation_info | dict | Yes | Execution limitation configuration |


## Predefined Workflow Types

Configuration files reference: [`python/ai_kernel_generator/config/`](../python/ai_kernel_generator/config/) directory.

### 1. Default Workflow ([default_workflow.yaml](../python/ai_kernel_generator/config/default_workflow.yaml))
**Flow**: `designer` → `coder` ←→ `verifier` → `finish`
- Complete design→coding→verification flow
- Can rollback to coder when verification fails

### 2. Fully Connected Workflow ([conductor_connect_all_workflow.yaml](../python/ai_kernel_generator/config/conductor_connect_all_workflow.yaml))  
**Flow**: Supports flexible transitions between all Agents
- Maximum flexibility, Agents can self-repair

### 3. Coder-Only Workflow ([coder_only_workflow.yaml](../python/ai_kernel_generator/config/coder_only_workflow.yaml))
**Flow**: `coder` ←→ `verifier` → `finish`
- Skip design phase, directly generate code

### 4. Verifier-Only Workflow ([verifier_only_workflow.yaml](../python/ai_kernel_generator/config/verifier_only_workflow.yaml))
**Flow**: `verifier` → `finish`
- Minimalist flow for verification only


## Agent Configuration Description

### Agent Information Structure
```yaml
agent_name:
  possible_next_agent: [list]         # List of possible next Agents
  output_format:                      # Output format definition (optional)
    parser_name: str                  # Parser name
    parser_definition: {...}          # Parser definition
```

## Decision Mechanisms

For detailed decision logic implementation, see [Conductor Design Document](./Conductor.md).

### Automatic Decision
- No available Agents → Directly finish
- Single available Agent and not mandatory analysis → Direct execution
- Parsing failed and retryable → Retry current Agent

### LLM Intelligent Decision
- Multiple available Agents → Intelligently select optimal path
- Single available Agent but in mandatory_llm_analysis list → Analyze and provide suggestions

Configuration files reference: [`python/ai_kernel_generator/config/`](../python/ai_kernel_generator/config/) directory for various workflow.yaml files.
