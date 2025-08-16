# Conductor Design Document

## Overview
Conductor is the intelligent task scheduler in the AI Kernel Generator. It inherits from `AgentBase` and manages the entire task execution flow based on workflow.yaml configuration. It uses LLM intelligent analysis of Agent execution results to decide the next execution flow and provide guidance.

## Core Functions
- **Intelligent Flow Scheduling**: Dynamically selects the next Agent based on workflow.yaml configuration and LLM intelligent analysis
- **Execution State Management**: Records and tracks all Agent execution results, maintaining complete task traces
- **Error Handling and Retry**: Intelligently handles Agent execution failures with automatic retry mechanisms
- **Flow Control**: Manages execution steps and repeat limitations to avoid infinite loops

## Initialization Parameters
| Parameter Name | Type/Required | Description |
|---------|---------|---------|
| op_name | str (Required) | Kernel name |
| task_desc | str (Required) | Task description |
| task_id | str (Required) | Task ID |
| dsl | str (Required) | DSL type: "triton", "swft", etc. |
| framework | str (Required) | Frontend framework: "mindspore", "torch", "numpy", etc. |
| arch | str (Required) | Hardware architecture: "ascend910b4", "a100", etc. |
| workflow_config_path | str (Optional) | Workflow configuration file path |
| config | dict (Required) | Complete configuration dictionary |

## Workflow Configuration
Conductor manages execution flow based on workflow.yaml configuration files. For detailed configuration documentation, see [Workflow Configuration](./Workflow.md).

### Configuration Elements
- **agent_info**: Agent flow definitions and output formats
- **start_agent**: Starting Agent
- **limitation_info**: Execution limits (maximum steps, repeat limits)
- **mandatory_llm_analysis**: List of Agents requiring mandatory LLM analysis

### Configuration Examples
Refer to detailed configuration examples:
- `config/default_workflow.yaml` - Standard Designer→Coder→Verifier flow
- `config/coder_only_workflow.yaml` - Simplified Coder+Verifier only flow
- `config/conductor_connect_all_workflow.yaml` - Fully connected Agent flow

## Key Methods

### get_next_agent() - Intelligent Decision Process
Executes four decision phases:
1. **State Update**: Increment step count, clear historical suggestions
2. **Retry Check**: Handle Agent parsing failure retry logic
3. **Option Retrieval**: Get valid options based on workflow configuration and current state
4. **Intelligent Decision**: Call LLM or execute directly based on option count and mandatory analysis configuration

### record_agent_execution() - Execution Recording
Records Agent execution results and updates task state:
- Save raw data to execution trace
- Parse results using appropriate parsers
- Update task information dictionary

### _llm_decide_next_agent() - LLM Intelligent Analysis
Uses LLM to analyze current execution state and decide the next Agent:
- Build analysis prompts based on `conductor/analyze.j2` template
- Comprehensively consider Agent results, error logs, valid options, etc.
- Return decided next Agent name and suggestions

## Typical Execution Flow
Based on default configuration:
```
designer → conductor(decision) → coder → conductor(decision) → verifier → conductor(intelligent analysis) → finish/coder
```

## Custom Extensions
Users can extend Conductor through the following approaches:

1. **Configuration Extension**: Modify workflow.yaml to define custom Agent flows
2. **Code Extension**: Override `get_next_agent()` to add custom decision logic
3. **Template Extension**: Modify `conductor/analyze.j2` to customize LLM analysis prompts
