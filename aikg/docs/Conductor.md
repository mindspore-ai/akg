# Conductor Design Document

## Overview
Conductor is the task commander component in the AI Kernel Generator. It inherits from `AgentBase` and is responsible for managing and coordinating the entire task execution flow. By checking the output results of various agents, it decides the next action to take and provides intelligent analysis and repair guidance when errors occur.

## Core Functions
- **Task Flow Management**: Coordinates the execution order of Designer, Coder, and Verifier.
- **Code Self-Verification**: Performs quality checks on the generated design documents and implementation code to decide whether to roll back for repairs.
- **Intelligent Error Analysis**: Analyzes the reasons for test failures and provides suggestions for fixes.
- **State Tracking and Recording**: Maintains a complete trace of task execution.
- **Retry Mechanism Control**: Manages the number of repair retries to avoid infinite loops.

## Initialization Parameters
| Parameter Name | Type/Required | Description |
|---------|---------|---------|
| op_name | str (Required) | Operation name, identifying the specific kernel. |
| task_id | str (Required) | Unique identifier for the task. |
| log_dir | str (Required) | Path to the log storage directory. |
| impl_type | str (Required) | Implementation type: "triton" or "swft". |
| model_config | dict (Required) | LLM model configuration, including `conductor_check` and `conductor_analyze` configurations. |

## Execution Flow get_next_action

1. **Trace Analysis Stage**
   - Get the record of the previous step (pre_trace).
   - Determine the execution path based on `action_type`.
   - Update the step counter.

2. **Decision Execution Stage**
   - After Designer/Coder completes: Execute `self_check()` for code inspection.
   - After Verifier fails: Execute `analyze_error()` for error analysis.
   - On success or exit condition met: Return `EXIT`.

3. **Result Return**
   - Returns a triplet: (next action type, parsed code object, repair suggestion).

## Key Method Descriptions

### self_check() - Code Self-Check
- **Function**: Checks the quality of the code output by Designer or Coder.
- **Process**: Parse code → Check retry limit → LLM performs quality assessment → Decide whether to perform a repair.
- **Output**: Next action (continue/repair) and suggestion (if a repair is needed).

### analyze_error() - Error Analysis  
- **Function**: Analyzes the root cause of a test failure.
- **Process**: Extract the most recent matching Designer and Coder code → LLM analyzes the error log → Pinpoint the source of the problem (Designer/Coder).
- **Output**: Repair target (Designer/Coder) and specific repair suggestions.

### initialize_check_docs() - Document Initialization
- **Function**: Initializes input data for various check templates based on `trace.base_doc`.
- **Supports**: Designer checks, Triton/SWFT Coder checks, error analysis.

## User-Defined Extensions

### Extension Overview
Referring to the AIKG project flowchart, the **Conductor module acts as the scheduling module**, centrally controlling the task flow and making decisions based on information stored in the `trace`. Users can freely modify or extend this module. **The main entry point for modification is the `get_next_action()` function**. Users can control the task flow according to their own designed flowcharts and judgment conditions.

### Default Execution Flow
Taking the typical task flow as an example:

- `DO_DESIGNER` → `conductor (self_check)` → `FIX_DESIGNER` / `DO_CODER`
- `DO_CODER` → `conductor (self_check)` → `FIX_CODER` / `VERIFY`  
- `VERIFY` → `conductor (error_analyze)` → `FIX_DESIGNER` / `FIX_CODER`

Through the Conductor, the various modules are connected to **form a self-checking loop** for intelligently analyzing tasks and generating kernels. 