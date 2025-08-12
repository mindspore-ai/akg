# Conductor Design Document

## Overview
Conductor is the task commander component in the AI Kernel Generator. It inherits from `AgentBase` and is responsible for managing and coordinating the entire task execution flow. Based on workflow.yaml configuration files, it provides intelligent workflow management by recording and analyzing the output results of various agents, making decisions on the next agent to execute, and providing intelligent guidance.

## Core Functions
- **Configuration-Based Workflow Management**: Dynamically manages agent execution flow based on workflow.yaml configuration
- **Intelligent Agent Decision Making**: Uses LLM to analyze current state and decide the next agent to execute
- **Execution Result Recording and Parsing**: Records all agent execution results and performs structured parsing
- **State Tracking and Trace Management**: Maintains complete task execution traces through Trace
- **Retry and Error Handling**: Intelligently handles parsing failures with agent retry mechanisms
- **Flow Control and Limitations**: Manages execution steps and repeat limitations to avoid infinite loops

## Initialization Parameters
| Parameter Name | Type/Required | Description |
|---------|---------|---------|
| op_name | str (Required) | Kernel name, identifying the specific kernel |
| task_desc | str (Required) | Task description, detailing the kernel functional requirements |
| task_id | str (Required) | Unique identifier for the task |
| dsl | str (Required) | Implementation type: "triton", "swft", etc. |
| framework | str (Required) | Frontend framework: "mindspore", "torch", "numpy", etc. |
| arch | str (Required) | Hardware architecture: "ascend910b4", "a100", etc. |
| workflow_config_path | str (Optional) | Workflow configuration file path, obtained from config if not provided |
| config | dict (Required) | Complete configuration dictionary, including log_dir, agent_model_config, etc. |

## Workflow Configuration System

### Configuration File Structure
Conductor manages the entire execution flow based on workflow.yaml configuration files, which mainly include:

- **agent_info**: Defines possible next steps and output formats for each agent
- **start_agent**: Specifies the starting agent
- **limitation_info**: Sets execution limits (maximum steps, repeat limits, etc.)
- **mandatory_analysis**: List of agents requiring mandatory LLM analysis

### Example Configuration
```yaml
agent_info:
  designer:
    possible_next_agent: [coder]
    output_format:
      parser_name: designer_parser
  coder:
    possible_next_agent: [verifier]
  verifier:
    possible_next_agent: [finish, coder]
start_agent: designer
mandatory_analysis: [verifier]
limitation_info:
  required:
    max_step: 20
```

## Execution Flow get_next_agent

1. **State Update Stage**
   - Increment step counter (step_count)
   - Clear previous conductor suggestions
   - Get current agent name

2. **Retry Check Stage**
   - Check if current agent parsing failed
   - If parsing failed and retryable, return same agent for retry

3. **Decision Execution Stage**
   - Get valid next agent options based on workflow configuration
   - Special handling for verifier results (success leads to finish, failure excludes finish option)
   - Decide whether LLM analysis is needed based on option count and mandatory_analysis configuration

4. **Intelligent Decision Making**
   - No options: directly finish
   - Single option and not mandatory analysis: direct execution
   - Single option with mandatory analysis or multiple options: call LLM for intelligent decision

## Key Method Descriptions

### record_agent_execution() - Agent Execution Recording
- **Function**: Records agent execution results, performs parsing and updates task information
- **Process**: Save raw data to trace → Parse results using appropriate parser → Update task_info
- **Parameters**: Agent name, execution result, prompt, reasoning process, error log, performance results
- **Returns**: Whether parsing was successful

### _llm_decide_next_agent() - LLM Intelligent Decision Making
- **Function**: Uses LLM to analyze current state and decide the next agent
- **Process**: Build input data → Call LLM → Parse decision results → Save suggestions
- **Template**: Uses conductor/analyze.j2 template for analysis
- **Input**: Current agent, agent results, error logs, valid options, etc.
- **Output**: Name of the decided next agent

### set_task_info() - Task Information Initialization
- **Function**: Initializes task information and base documents based on workflow configuration
- **Support**: Dynamic field initialization, base document integration

## User-Defined Extensions

### Extension Overview
**The Conductor module serves as an intelligent scheduling center**, controlling task flow based on workflow.yaml configuration files and LLM intelligent analysis. Users can customize extensions through the following approaches:

1. **Configuration File Extension**: Modify or create new workflow.yaml configuration files to define custom agent flows
2. **Code Extension**: Modify the `get_next_agent()` function to add custom decision logic
3. **Template Extension**: Modify the conductor/analyze.j2 template to customize LLM analysis prompts

### Configuration File Extension Example
Users can create custom workflow configurations, such as adding new agents or modifying flows:

```yaml
agent_info:
  designer:
    possible_next_agent: [coder, designer]  # Support designer self-repair
  coder:
    possible_next_agent: [verifier, coder]
  verifier:
    possible_next_agent: [finish, designer, coder, optimizer]  # Add optimizer agent
  optimizer:
    possible_next_agent: [verifier, optimizer]
start_agent: designer
mandatory_analysis: [verifier, optimizer]  # Force LLM analysis for critical agents
```

### Typical Execution Flow
Based on default configuration:

- `designer` → `conductor (decision)` → `coder`
- `coder` → `conductor (decision)` → `verifier`  
- `verifier` → `conductor (intelligent analysis)` → `finish` / `coder`

Through Conductor's intelligent scheduling, agents form **adaptive execution loops**, dynamically selecting optimal paths based on task state.

### Extension Key Points
- **Main Extension Entry**: `get_next_agent()` function and workflow.yaml configuration files
- **Decision Logic**: Custom LLM decision flow can be implemented in `_llm_decide_next_agent()`
- **State Management**: Access complete execution state through `task_info` and `trace`
- **Parser Extension**: Support result parsing for new agents through `record_agent_execution()` 