# 默认DSL配置文档

## 概述

DefaultDSLConfig是AI Kernel Generator中针对不同DSL（Domain Specific Language）的默认配置管理系统。它为每种DSL（如Triton、SWFT等）提供预设的配置模板，包括模型配置、文档路径、工作流设置等，简化用户的配置过程。

## 配置文件结构

默认DSL配置文件采用YAML格式，主要包含以下配置项：

```yaml
# 代理模型配置
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
verify_timeout: 300  # 验证超时时间（秒）
```

## 主要配置项说明

### 1. 代理模型配置 (agent_model_config)

定义各个代理使用的LLM模型：

| 代理名称 | 类型 | 描述 |
|---------|------|------|
| designer | str | 设计器代理使用的模型 |
| coder | str | 编码器代理使用的模型 |
| conductor | str | 指挥者代理使用的模型 |
| api_generator | str | API生成器代理使用的模型 |

**示例配置**：
```yaml
agent_model_config:
  designer: deepseek_r1_default
  coder: deepseek_r1_default
  conductor: deepseek_r1_default
  api_generator: deepseek_r1_default
```

### 2. 日志配置 (log_dir)

指定任务执行日志的存储目录：

- **格式**: 字符串路径
- **支持**: 绝对路径和相对路径（支持`~`表示用户主目录）
- **默认**: `"~/aikg_logs"`

### 3. 工作流配置 (workflow_config_path)

指定该DSL使用的默认工作流配置文件：

- **格式**: 相对于项目根目录的路径
- **用途**: 定义代理执行流程和限制
- **示例**: `"config/default_workflow.yaml"`

### 4. 文档目录配置 (docs_dir)

为不同代理指定参考文档的目录：

```yaml
docs_dir:
  designer: "resources/docs/aul_docs"    # 设计器文档（如AUL规范）
  coder: "resources/docs/triton_docs"    # 编码器文档（如DSL语法）
```

**用途**：
- Designer: 提供算法设计规范文档
- Coder: 提供目标DSL的语法和API文档

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
- **默认**: 300秒（5分钟）
- **用途**: 防止验证过程无限等待

## 预定义DSL配置

### Triton配置 (default_triton_config.yaml)

**适用场景**: Triton kernel开发

**特点**:
- 使用 AUL / Sketch 作为设计语言
- 目标生成 Triton 代码
- 支持 Ascend NPU / CUDA GPU 后端

**配置示例**: [`python/ai_kernel_generator/config/default_triton_config.yaml`](../../python/ai_kernel_generator/config/default_triton_config.yaml)

### SWFT配置 (default_swft_config.yaml)

**适用场景**: 华为Ascend NPU上的SWFT kernel开发

**特点**:
- 使用AUL作为设计语言
- 目标生成SWFT代码
- 支持Ascend NPU后端

## 配置使用方法

### 1. 直接使用预定义配置

```python
from ai_kernel_generator import load_config

# 加载Triton默认配置
config = load_config("default_triton_config")

# 创建任务
task = Task(
    op_name="relu",
    task_desc="...",
    dsl="triton",
    config=config
)
```

### 2. 自定义配置覆盖

```python
# 基于默认配置进行自定义
config = load_config("default_triton_config")

# 覆盖特定配置项
config.update({
    "log_dir": "/custom/log/path",
    "agent_model_config": {
        "designer": "custom_model",
        "coder": "custom_model"
    }
})
```

### 3. 创建新的DSL配置

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

## 配置扩展指南

### 1. 添加新的代理配置

```yaml
agent_model_config:
  designer: model_name
  coder: model_name
  conductor: model_name
  optimizer: model_name    # 新增优化器代理
```

### 2. 扩展文档目录配置

```yaml
docs_dir:
  designer: "path/to/design/docs"
  coder: "path/to/code/docs"
  optimizer: "path/to/optimization/docs"  # 新增优化文档
```

## 最佳实践

### 配置管理原则

1. **版本控制**: 将配置文件纳入版本控制，便于追踪变更
2. **环境分离**: 为开发、测试、生产环境使用不同的配置
3. **模块化设计**: 将通用配置和特定配置分离
4. **文档同步**: 配置变更时及时更新相关文档

### 性能优化建议

1. **模型选择**: 根据任务复杂度选择合适的LLM模型
2. **日志管理**: 合理设置日志级别，避免产生过多日志文件
3. **超时设置**: 根据硬件性能调整验证超时时间
4. **性能测试**: 适当调整profile_settings参数，平衡精度和效率

### 故障排除

1. **路径问题**: 确保所有路径配置正确且文件存在
2. **模型配置**: 验证LLM模型名称和API配置正确
3. **权限问题**: 确保日志目录有写入权限
4. **内存管理**: 监控性能测试时的内存使用情况

通过DefaultDSLConfig系统，AI Kernel Generator为不同DSL提供了标准化的配置管理，简化了用户配置过程，提高了系统的易用性和可维护性。
