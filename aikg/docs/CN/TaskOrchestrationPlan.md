# 任务编排方案配置（Task Orchestration Plan Configuration）

## 概述

`任务编排方案配置（简称 编排配置）`用于声明一次算子生成任务的完整运行方案，由 `Task` 在运行时加载，用于驱动各 Agent 的协作与落地。

编排配置主要包含：
- `agent_model_config`：为各 Agent 指定大模型预设（取值来自 `core/llm/llm_config.yaml`）。
- `workflow_config_path`：指向工作流 YAML，用于定义执行流程（详见《[工作流系统设计文档](./Workflow.md)》）。
- `docs_dir`：为各 Agent 提供参考文档目录（详见《[文档驱动式接入指南](./DocDrivenIntegration.md)》）。
- `log_dir`：任务日志根目录。
- `profile_settings`：性能测试参数（如 `run_times`、`warmup_times`）。
- `verify_timeout`：验证超时时间（单位：秒）。

## 配置文件结构

`编排配置` 采用 YAML 格式，主要包含以下配置项：

```yaml
# Agent模型配置
agent_model_config:
  designer: model_name
  coder: model_name
  conductor: model_name
  api_generator: model_name

# 日志配置
log_dir: "~/aikg_logs"

# 工作流配置
workflow_config_path: "config/workflow_file.yaml"

# 文档目录配置
docs_dir:
  designer: "path/to/designer/docs"
  coder: "path/to/coder/docs"

# 性能分析配置
profile_settings:
  run_times: 50
  warmup_times: 5

# 验证配置
verify_timeout: 300  # 验证超时时间，单位秒
```

## 主要配置项

### 1. Agent模型配置 (agent_model_config)

`agent_model_config` 用于为每个 Agent 指定所用的大模型预设；取值必须来自 `core/llm/llm_config.yaml`（例如：`deepseek_r1_default`）。
常见键：`designer`、`coder`、`conductor`、`api_generator`（可按需扩展）。如名称不在预设中，会在校验时报错。

```yaml
# 示例Agent模型配置
agent_model_config:
  designer: deepseek_r1_default
  coder: deepseek_r1_default
```

### 2. 日志配置 (log_dir)

指定任务执行日志的存储目录：

- **格式**: 字符串路径
- **支持**: 绝对路径和相对路径（支持`~`表示用户主目录）
- **默认值**: `"~/aikg_logs"`
- **最终路径形态**: 实际写入时会在该目录下自动创建形如 `Task_{随机ID}` 的子目录。

### 3. 工作流配置 (workflow_config_path)

- 指向本次任务的工作流 YAML，用于配置详细的 AIKG 任务工作流程。示例：`"config/default_workflow.yaml"`。
- 详见《[工作流系统设计文档](./Workflow.md)》。

### 4. 文档目录配置 (docs_dir)

详见《[文档驱动式接入指南](./DocDrivenIntegration.md)》。

```yaml
# 示例文档配置
docs_dir:
  designer: "resources/docs/triton_docs"    # Designer文档（如DSL语法）
  coder: "resources/docs/triton_docs"    # Coder文档（如DSL语法）
```


### 5. 性能分析配置 (profile_settings)

配置性能测试的执行参数：

```yaml
profile_settings:
  run_times: 50      # 性能测试运行次数
  warmup_times: 5    # 预热运行次数
```

### 6. 验证配置 (verify_timeout)

设置代码验证的超时时间：

- **单位**: 秒
- **默认值**: 300秒（5分钟）
- **作用**: 防止验证过程无限等待

## 预设方案配置

### Triton配置 (default_triton_config.yaml)

**适用场景**: Triton kernel开发

**特点**:
- 目标 Triton 代码生成
- 支持 Ascend NPU / CUDA GPU 后端
- 在 coder-only 工作流下，Designer 不启用

**配置示例**: [`config/default_triton_config.yaml`](../../python/ai_kernel_generator/config/default_triton_config.yaml)

### SWFT配置 (default_swft_config.yaml)

**适用场景**: 华为昇腾NPU上的SWFT kernel开发

**特点**:
- 目标 SWFT 代码生成
- 支持昇腾 NPU 后端
- 在 coder-only 工作流下，Designer 不启用

**配置示例**: [`config/default_swft_config.yaml`](../../python/ai_kernel_generator/config/default_swft_config.yaml)

## 配置使用方法

```python
from ai_kernel_generator.config.config_validator import load_config

# 方式1：按 DSL 加载默认方案
config = load_config(dsl="triton")

# 方式2：显式指定方案文件路径
config = load_config(config_path="python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

# 创建任务
task = Task(op_name="relu", task_desc="...", dsl="triton", config=config)
```


### 3. 创建新的方案文件

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

## 配置扩展指南

### 1. 添加新Agent配置

```yaml
agent_model_config:
  designer: model_name
  coder: model_name
  conductor: model_name
  optimizer: model_name    # 新增优化器agent
```

### 2. 扩展文档目录配置

```yaml
docs_dir:
  designer: "path/to/design/docs"
  coder: "path/to/code/docs"
  optimizer: "path/to/optimization/docs"  # 新增优化文档
```
