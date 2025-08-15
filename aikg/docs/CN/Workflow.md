# Workflow 工作流系统设计文档

## 概述

Workflow是AIKG的核心配置管理系统，通过YAML配置文件定义和控制代理（Agent）的执行流程。它为Conductor提供智能调度的规则基础，支持灵活的工作流定制，实现不同场景下的最优执行路径。

## 核心理念

- **配置驱动**：通过YAML文件声明式地定义工作流，降低代码耦合度
- **智能决策**：结合LLM分析和规则约束，实现自适应的代理调度
- **模块化设计**：支持代理的灵活组合，满足不同任务需求
- **可扩展性**：便于添加新代理和自定义执行逻辑
- **约束管理**：内置重试限制和步数控制，确保系统稳定性

## 工作流配置结构

### 配置文件格式

```yaml
# 代理信息定义
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
            field_description: "字段描述"

# 起始代理
start_agent: agent_name

# 强制LLM分析的代理列表（由此agent进入conductor后，conductor强制使用LLM分析）
mandatory_llm_analysis: [agent1, agent2]

# 限制信息
limitation_info:
  required: # 必填限制信息
    max_step: 20  # 最大执行步数
  optional: # 可选限制信息
    repeat_limits:
      single_agent:
        agent_name: 3  # 单代理最大连续重复次数
      sequences:
        sequence_name:
          pattern: [agent1, agent2]  # 序列模式
          max_repeats: 3             # 序列最大重复次数
```

### 核心配置字段

| 字段名称 | 类型 | 必选 | 描述 |
|---------|------|------|------|
| agent_info | dict | 是 | 定义所有代理的配置信息 |
| start_agent | str | 是 | 指定工作流的起始代理 |
| mandatory_llm_analysis | list | 否 | 需要强制LLM分析的代理列表 |
| limitation_info | dict | 是 | 执行限制配置 |

## 预定义工作流类型

具体的工作流配置文件可以参考：
- [`python/ai_kernel_generator/config/default_workflow.yaml`](../../python/ai_kernel_generator/config/default_workflow.yaml)
- [`python/ai_kernel_generator/config/conductor_connect_all_workflow.yaml`](../../python/ai_kernel_generator/config/conductor_connect_all_workflow.yaml)
- [`python/ai_kernel_generator/config/coder_only_workflow.yaml`](../../python/ai_kernel_generator/config/coder_only_workflow.yaml)
- [`python/ai_kernel_generator/config/verifier_only_workflow.yaml`](../../python/ai_kernel_generator/config/verifier_only_workflow.yaml)

### 1. 默认工作流 (default_workflow.yaml)
**适用场景**：标准的算子生成流程，适合大多数场景

**流程**：`designer` → `coder` ←→ `verifier` → `finish`

**特点**：
- 线性流程，每个阶段专注单一任务
- 验证失败时支持回退到coder
- 适合算法明确、需要完整设计文档的场景

### 2. 全连接工作流 (conductor_connect_all_workflow.yaml)
**适用场景**：复杂算子，需要多轮迭代优化

**流程**：支持所有代理间的灵活跳转

**特点**：
- 最大灵活性，支持任意代理间跳转
- 设计器和编码器可自我修复
- 适合复杂算法和多轮优化场景

### 3. 仅编码工作流 (coder_only_workflow.yaml)
**适用场景**：算法设计明确，只需代码实现

**流程**：`coder` ←→ `verifier` → `finish`

**特点**：
- 跳过设计阶段，直接生成代码
- 编码器可自我修复
- 适合标准算子或参考实现充足的场景

### 4. 仅验证工作流 (verifier_only_workflow.yaml)
**适用场景**：代码已存在，只需验证

**流程**：`verifier` → `finish`

**特点**：
- 最简化流程，仅用于验证
- 适合代码质量检查和性能测试

## 代理配置详解

### 代理信息结构

```yaml
agent_name:
  possible_next_agent: [list]  # 可能的下一个代理列表
  output_format:               # 输出格式定义（可选）
    parser_name: str           # 解析器名称
    parser_definition:         # 解析器定义
      output_fields:           # 输出字段定义
        field_name:
          field_type: str      # 字段类型
          mandatory: bool      # 是否必填
          field_description: str # 字段描述
```

### 特殊代理说明

#### Designer（设计器）
- **职责**：生成算法伪代码或实现草图
- **输出格式**：需要解析器处理代码字段
- **典型下一步**：coder（代码实现）

#### Coder（编码器）
- **职责**：将设计转换为具体的实现代码
- **输出格式**：需要解析器处理代码字段
- **典型下一步**：verifier（验证）

#### Verifier（验证器）
- **职责**：验证代码正确性和性能
- **输出格式**：程序化验证，无需解析器
- **典型下一步**：finish（完成）或其他代理（修复）

## 限制机制说明

### 执行步数限制
```yaml
limitation_info:
  required:
    max_step: 20  # 防止无限循环
```

### 重复限制
```yaml
limitation_info:
  optional:
    repeat_limits:
      # 单代理连续重复限制
      single_agent:
        designer: 2  # designer最多连续执行2次
        coder: 2     # coder最多连续执行2次
      
      # 序列重复限制
      sequences:
        coder_verifier:
          pattern: [coder, verifier]  # coder->verifier序列
          max_repeats: 3              # 该序列最多重复3次
```

## 决策机制

### 自动决策条件
1. **无可选代理**：直接结束
2. **单一可选代理且非强制分析**：直接执行
3. **解析失败且可重试**：重试当前代理

### LLM智能决策条件
1. **多个可选代理**：需要智能选择
2. **单一可选代理但在mandatory_llm_analysis列表中**：需要分析建议

## 自定义工作流指南

### 1. 创建新工作流

```yaml
# custom_workflow.yaml
agent_info:
  # 定义自定义代理或修改现有代理流程
  custom_optimizer:
    possible_next_agent: [verifier, custom_optimizer]
    output_format:
      parser_name: optimizer_parser
      parser_definition:
        output_fields:
          optimized_code:
            field_type: str
            mandatory: true
            field_description: "优化后的代码"

start_agent: custom_optimizer

mandatory_llm_analysis: [custom_optimizer]

limitation_info:
  required:
    max_step: 15
```

### 2. 使用自定义工作流

```python
# 在Task初始化时指定自定义工作流
task = Task(
    op_name="custom_op",
    task_desc="...",
    workflow="custom_workflow"  # 使用自定义工作流
)
```

### 3. 扩展现有工作流

```yaml
# 基于default_workflow扩展
agent_info:
  designer:
    possible_next_agent: [coder]  # 添加optimizer选项
  coder:
    possible_next_agent: [verifier]
  verifier:
    possible_next_agent: [finish, coder, optimizer]
  optimizer:  # 新增optimizer代理
    possible_next_agent: [verifier, optimizer]

start_agent: designer
mandatory_llm_analysis: [optimizer]  # 强制分析optimizer
```

## 最佳实践

### 工作流设计原则

1. **明确目标**：根据具体使用场景选择合适的工作流类型
2. **最小化复杂度**：优先使用简单的线性流程
3. **合理设置限制**：避免无限循环，设置合适的重试次数
4. **强制分析配置**：对关键决策点配置mandatory_llm_analysis

### 配置建议

1. **开发阶段**：使用全连接工作流，允许灵活调试
2. **生产环境**：使用受限工作流，确保稳定性
3. **性能优化**：使用coder_only工作流，专注代码质量
4. **质量检查**：使用verifier_only工作流，验证现有代码

### 扩展指导

1. **新代理开发**：
   - 定义清晰的输入输出格式
   - 实现相应的解析器
   - 在工作流中配置合适的路径

2. **工作流优化**：
   - 根据实际使用情况调整限制参数
   - 分析代理执行历史，优化决策路径
   - 使用mandatory_llm_analysis控制关键决策点

3. **调试支持**：
   - 通过Trace查看完整执行轨迹
   - 分析Conductor决策日志
   - 调整工作流配置验证效果

通过Workflow系统，AI Kernel Generator实现了灵活、可控、可扩展的任务执行流程管理，为不同场景提供了最优的代理协作方案。
