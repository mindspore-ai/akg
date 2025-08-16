# Conductor 设计文档

## 概述
Conductor 是 AI Kernel Generator 中的智能任务调度器，继承自 `AgentBase`，负责基于 workflow.yaml 配置管理整个任务执行流程。它通过LLM智能分析各Agent执行结果，决策下一步执行流程并提供指导建议。

## 核心功能
- **智能流程调度**：基于 workflow.yaml 配置和LLM智能分析动态选择下一个执行的Agent
- **执行状态管理**：记录和跟踪所有Agent执行结果，维护完整的任务轨迹
- **错误处理与重试**：智能处理Agent执行失败，支持自动重试机制
- **流程控制**：管理执行步数和重复限制，避免无限循环

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称 |
| task_desc | str (必选) | 任务描述 |
| task_id | str (必选) | 任务ID |
| dsl | str (必选) | DSL类型："triton"、"swft"等 |
| framework | str (必选) | 前端框架："mindspore"、"torch"、"numpy"等 |
| arch | str (必选) | 硬件架构："ascend910b4"、"a100"等 |
| workflow_config_path | str (可选) | workflow配置文件路径 |
| config | dict (必选) | 完整配置字典 |

## 工作流配置
Conductor基于 workflow.yaml 配置文件管理执行流程，详细的配置说明参见 [Workflow配置文档](./Workflow.md)。

### 配置要素
- **agent_info**: Agent流程定义和输出格式
- **start_agent**: 起始Agent
- **limitation_info**: 执行限制（最大步数、重复限制）
- **mandatory_llm_analysis**: 强制LLM分析的Agent列表

### 配置示例
详细配置示例可参考：
- `config/default_workflow.yaml` - 标准Designer→Coder→Verifier流程
- `config/coder_only_workflow.yaml` - 仅Coder+Verifier的简化流程
- `config/conductor_connect_all_workflow.yaml` - 全连接Agent流程

## 核心方法

### get_next_agent() - 智能决策流程
执行四个阶段的决策：
1. **状态更新**：增加步骤计数，清理历史建议
2. **重试检查**：处理Agent解析失败的重试逻辑
3. **选项获取**：基于workflow配置和当前状态获取有效选项
4. **智能决策**：根据选项数量和强制分析配置调用LLM或直接执行

### record_agent_execution() - 执行记录
记录Agent执行结果并更新任务状态：
- 保存原始数据到执行轨迹
- 使用相应解析器解析结果
- 更新任务信息字典

### _llm_decide_next_agent() - LLM智能分析
使用LLM分析当前执行状态并决策下一个Agent：
- 基于 `conductor/analyze.j2` 模板构建分析提示
- 综合考虑Agent结果、错误日志、有效选项等信息
- 返回决策的下一个Agent名称和建议

## 典型执行流程
基于默认配置的执行流程：
```
designer → conductor(决策) → coder → conductor(决策) → verifier → conductor(智能分析) → finish/coder
```

## 自定义扩展
用户可通过以下方式扩展Conductor：

1. **配置扩展**：修改workflow.yaml定义自定义Agent流程
2. **代码扩展**：重写`get_next_agent()`添加自定义决策逻辑  
3. **模板扩展**：修改`conductor/analyze.j2`定制LLM分析提示
