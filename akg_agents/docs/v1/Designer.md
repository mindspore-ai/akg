# Designer Design Document

## Overview
Designer is a core component in the AI Kernel Generator that automatically generates algorithm design documents based on Large Language Models (LLMs). It inherits from `AgentBase` and is responsible for intelligently generating high-quality algorithm sketches based on the kernel name, task description, and hardware configuration. The Designer uses AUL (AI Unity Language) or similar design languages to express algorithm logic.

## Core Functions
- **Intelligent Design Generation**: Automatically generates algorithm design documents based on kernel requirements
- **Doc-Driven Integration**: Supports custom reference documents to improve generation quality
- **Multi-DSL Support**: Supports different design languages
- **Hardware-Aware Design**: Considers hardware characteristics during design generation
- **Document Integration**: Automatically loads design specifications and reference materials

## Initialization Parameters
| Parameter Name | Type/Required | Description |
|---------|---------|---------|
| op_name | str (Required) | Kernel name, identifying the specific kernel |
| task_desc | str (Required) | Task description, detailing the kernel functional requirements |
| dsl | str (Required) | Design language: "triton_cuda", "triton_ascend", "swft", etc. |
| backend | str (Required) | Hardware backend: "ascend", "cuda", etc. |
| arch | str (Required) | Hardware architecture: "ascend910b4", "a100", etc. |
| workflow_config_path | str (Optional) | Workflow configuration file path (usually injected by Task from the orchestration plan) |
| config | dict (Required) | Complete orchestration plan configuration including log_dir, agent_model_config, docs_dir, etc. (see [Task Orchestration Plan Configuration](./TaskOrchestrationPlan.md)) |

> Related docs: Workflow details in [Workflow](./Workflow.md); documentation integration in [Doc-Driven Integration Guide](./DocDrivenIntegration.md).

## Doc-Driven Integration

The Designer loads reference documents via `docs_dir.designer` in the orchestration plan. For document list and specifications, see the [Doc-Driven Integration Guide](./DocDrivenIntegration.md). Details are not duplicated here.

## Execution Flow

1. **Initialization Stage**
   - Load workflow configuration and create parser
   - Initialize design generation template
   - Load reference documents using Doc-Driven Integration
   - Prepare base document structure

2. **Generation Stage**
   - Process task information and conductor suggestions
   - Execute LLM generation using loaded documents
   - Return generated design, prompt, and reasoning

3. **Document Structure**
   - DSL specifications and syntax rules
   - Algorithm design patterns and examples
   - Hardware-specific considerations
   - Format instructions for output parsing
