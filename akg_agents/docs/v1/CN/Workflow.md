# Workflow 工作流系统设计文档

## 概述
Workflow是AIKG的核心配置系统，通过YAML配置文件定义Agent执行流程，为Conductor提供智能调度规则，支持灵活的工作流定制。


## 配置文件结构

### 基本格式
```yaml
# Agent信息定义
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

# 起始Agent
start_agent: agent_name

# 强制LLM分析的代理列表（由此agent进入conductor后，conductor强制使用LLM分析）
mandatory_llm_analysis: [agent1, agent2]

# 执行限制
limitation_info:
  required: # 必填限制信息
    max_step: 20  # 最大执行步数
  optional: # 可选限制信息
    repeat_limits:
      single_agent:
        agent_name: 3                 # 单Agent最大连续重复次数
      sequences:
        sequence_name:
          pattern: [agent1, agent2]   # 序列模式
          max_repeats: 3              # 序列最大重复次数
```

### 核心配置字段
| 字段名称 | 类型 | 必选 | 描述 |
|---------|------|------|------|
| agent_info | dict | 是 | 定义所有Agent的配置信息 |
| start_agent | str | 是 | 指定起始Agent |
| mandatory_llm_analysis | list | 否 | 需要强制LLM分析的Agent列表 |
| limitation_info | dict | 是 | 执行限制配置 |


## 预定义工作流类型

配置文件参考：[`python/ai_kernel_generator/config/`](../../python/ai_kernel_generator/config/) 目录。

### 1. 标准工作流 ([default_workflow.yaml](../../python/ai_kernel_generator/config/default_workflow.yaml))
**流程**: `designer` → `coder` ←→ `verifier` → `finish`
- 完整的设计→编码→验证流程
- 验证失败时可回退到coder

### 2. 全连接工作流 ([conductor_connect_all_workflow.yaml](../../python/ai_kernel_generator/config/conductor_connect_all_workflow.yaml))  
**流程**: 支持所有Agent间灵活跳转
- 最大灵活性，Agent可自我修复

### 3. 仅编码工作流 ([coder_only_workflow.yaml](../../python/ai_kernel_generator/config/coder_only_workflow.yaml))
**流程**: `coder` ←→ `verifier` → `finish`
- 跳过设计阶段，直接生成代码

### 4. 仅验证工作流 ([verifier_only_workflow.yaml](../../python/ai_kernel_generator/config/verifier_only_workflow.yaml))
**流程**: `verifier` → `finish`
- 最简化流程，仅用于验证


## Agent配置说明

### Agent信息结构
```yaml
agent_name:
  possible_next_agent: [list]         # 可能的下一个Agent列表
  output_format:                      # 输出格式定义（可选）
    parser_name: str                  # 解析器名称
    parser_definition: {...}          # 解析器定义
```

## 决策机制

详细的决策逻辑实现参见 [Conductor设计文档](./Conductor.md)。

### 自动决策
- 无可选Agent → 直接结束
- 单一可选Agent且非强制分析 → 直接执行
- 解析失败且可重试 → 重试当前Agent

### LLM智能决策
- 多个可选Agent → 智能选择最优路径
- 单一可选Agent但在mandatory_llm_analysis列表中 → 分析并提供建议

配置文件参考：[`python/ai_kernel_generator/config/`](../../python/ai_kernel_generator/config/) 目录下的各种workflow.yaml文件。
