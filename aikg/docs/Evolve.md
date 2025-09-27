# AI Kernel Generator Evolution Script Usage Guide

## Overview

AI Kernel Generator provides four files for automatic kernel generation and optimization:

1. **`evolve.py`** - Core evolution algorithm implementation
2. **`single_evolve_runner.py`** - Single task evolution executor
3. **`run_batch_evolve.py`** - Batch task evolution executor
4. **`run_torch_evolve_triton.py`** - PyTorch + Triton evolution example script

Related configuration can be found in the **`evolve_config.yaml`** configuration file

## Feature Description

### 1. evolve.py - Core Evolution Algorithm

The evolve.py file implements a genetic algorithm-based kernel evolution generation system, supporting two operation modes:

- **Island Mode**: Maintains population diversity through a distributed island model, ensures excellent individuals are not lost through an elite retention mechanism, enables information sharing between islands through migration mechanisms, and combines stratified sampling mechanisms to ensure diversity and innovation in the evolution process
- **Simple Mode**: A simplified evolution mode suitable for quick testing and simple tasks, without complex island management and elite mechanisms

Both modes significantly improve kernel generation success rate and quality, allowing users to choose the appropriate mode based on task complexity.

**Mode Selection**

The system automatically selects the operation mode based on parameters:
- **Island Mode** is enabled when `num_islands > 1` and `elite_size > 0`
- **Simple Mode** is used otherwise

**Main Process**

1. **Initialization**
   - Create storage directories
   - Configure parameters based on selected mode
   - Load meta-prompts

2. **Evolution Loop**
   Each round executes:
   - **Migration**: Exchange elite individuals between islands (Island Mode only)
   - **Inspiration Sampling**: Select parent generation and sample historical implementations
     - Island Mode: Sample within specified islands with elite pool support
     - Simple Mode: Sample from global historical implementations
   - **Task Execution**: Generate and test multiple implementations in parallel
   - **Result Processing**: Collect successful implementations and generate sketch descriptions
   - **Elite Management**: Update elite library and save best implementations (Island Mode only)

3. **Final Processing**
   - Sort all best implementations
   - Calculate statistical information
   - Return evolution results

**Initialization Parameters**:
| Parameter Name | Type/Required | Parameter Description |
|----------------|---------------|----------------------|
| op_name | str (Required) | Operator name |
| task_desc | str (Required) | Task description |
| dsl | str (Required) | DSL type: "triton", "swft", etc. |
| framework | str (Required) | Frontend framework: "mindspore", "torch", "numpy", etc. |
| backend | str (Required) | Backend type: "ascend", "cuda", etc. |
| arch | str (Required) | Hardware architecture: "a100", "ascend910b4", etc. |
| max_rounds | int (Required) | Maximum evolution rounds |
| parallel_num | int (Required) | Parallel tasks per round |
| num_islands | int (Required) | Number of islands |
| migration_interval | int (Required) | Migration interval |
| elite_size | int (Required) | Elite retention size |
| parent_selection_prob | float (Required) | Parent selection probability |
| device_list | List[int] (Required) | Device list |

#### Island Mode

The island mode is a distributed evolutionary algorithm that divides the evolutionary population into multiple independent "islands". Each island evolves independently while enabling information exchange between islands through migration mechanisms. By maintaining population diversity, exploring different solution space regions, and providing system robustness, it improves the success rate and quality of kernel generation.

**Elite Retention and Parent Selection Mechanism**: Provides high-quality parent selection for subsequent rounds. By maintaining a fixed-size elite pool sorted by performance, combined with probability control for island selection and elite pool selection, it avoids over-reliance on a single source, increases implementation diversity, and ensures stability and effectiveness of the evolution process.

**Configuration Example**
```yaml
# Enable Island Mode
island:
  num_islands: 2              # Create 2 islands
  migration_interval: 2       # Perform migration every 2 rounds
  elite_size: 5              # Elite pool size
  parent_selection_prob: 0.7  # 70% probability to select from current island, 30% from elite pool
```

#### Simple Mode

Simple mode is a simplified evolution mode suitable for quick testing and simple tasks. It does not require complex island management and elite mechanisms, directly using global historical implementations for sampling and evolution.

**Configuration Example**
```yaml
# Enable Simple Mode
island:
  num_islands: 1              # Set to 1 to disable island mode
  elite_size: 0              # Set to 0 to disable elite mechanism
```

#### Stratified Sampling Mechanism

The stratified sampling mechanism intelligently selects inspiration from historical implementations, providing diverse reference samples for each round of evolution. By categorizing implementations into good, medium, and poor performance levels and sampling from different levels to avoid selecting only the optimal individuals, combined with a repetition avoidance mechanism to exclude implementations already generated in the current round, it ensures diversity and innovation in the evolution process.

This mechanism works in both Island Mode and Simple Mode:
- **Island Mode**: Performs stratified sampling within historical implementations of the specified island
- **Simple Mode**: Performs stratified sampling within global historical implementations


### 2. single_evolve_runner.py - Single Task Executor

single_evolve_runner.py is a single task evolution executor used to execute evolution generation tasks for individual operators. It supports both configuration file and command-line parameter configuration methods, provides detailed execution logs and result statistics, and is an ideal tool for testing and debugging individual operators.

**Usage**:
```bash
# Use default configuration
python single_evolve_runner.py

# Use specified configuration file
python single_evolve_runner.py evolve_config.yaml
```

### 3. run_batch_evolve.py - Batch Executor

run_batch_evolve.py is a batch task evolution executor used to execute evolution generation tasks for multiple operators in batch. It supports parallel execution and dynamic device allocation, provides detailed execution statistics and report generation, and is an efficient tool for large-scale operator generation in production environments.

**Usage**:
```bash
# Use default configuration
python run_batch_evolve.py

# Use specified configuration file
python run_batch_evolve.py evolve_config.yaml
```

## Usage Examples

### 4. PyTorch + Triton Evolution Example

The `run_torch_evolve_triton.py` example demonstrates how to use the evolution system with PyTorch framework and Triton DSL for automatic kernel generation and optimization.

**Key Features**:
- **Framework**: PyTorch with CUDA backend
- **DSL**: Triton for GPU kernel generation
- **Architecture**: A100 GPU support
- **Evolution Mode**: Island mode with 2 islands and elite retention
- **Task**: ReLU activation function optimization

**Usage**:
```bash
# Run PyTorch evolution example
python run_torch_evolve_triton.py
```
