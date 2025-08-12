# Conductor 设计文档

## 概述
Conductor 是 AI Kernel Generator 中的任务指挥者组件，继承自 `AgentBase`，负责管理和协调整个任务执行流程。它基于workflow.yaml配置文件进行智能工作流管理，通过记录和分析各个代理的输出结果，决策下一步的执行代理，并提供智能建议指导。

## 核心功能
- **基于配置的工作流管理**：根据workflow.yaml配置动态管理代理执行流程
- **智能代理决策**：使用LLM分析当前状态并决策下一个执行的代理
- **执行结果记录与解析**：记录所有代理执行结果并进行结构化解析
- **状态跟踪与轨迹管理**：通过Trace维护完整的任务执行轨迹
- **重试与错误处理**：智能处理解析失败，支持代理重试机制
- **流程控制与限制**：管理执行步数、重复限制，避免无限循环

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称，标识具体的算子 |
| task_desc | str (必选) | 任务描述，详细说明算子功能需求 |
| task_id | str (必选) | 任务唯一标识符 |
| dsl | str (必选) | 实现类型："triton"、"swft"等DSL |
| framework | str (必选) | 前端框架："mindspore"、"torch"、"numpy"等 |
| arch | str (必选) | 硬件架构："ascend910b4"、"a100"等 |
| workflow_config_path | str (可选) | workflow配置文件路径，如不提供则从default_{dsl}_config.yaml中获取 |
| config | dict (必选) | 完整配置字典，包含log_dir、agent_model_config等 |

## 工作流配置体系

### 配置文件结构
Conductor基于workflow.yaml配置文件管理整个执行流程，主要包含：

- **agent_info**: 定义各代理的可能下一步和输出格式
- **start_agent**: 指定起始代理
- **limitation_info**: 设置执行限制（最大步数、重复限制等）
- **mandatory_analysis**: 需要强制LLM分析的代理列表

### 示例配置
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

## 执行流程 get_next_agent

1. **状态更新阶段**
   - 增加步骤计数器（step_count）
   - 清除上一次的conductor建议
   - 获取当前代理名称

2. **重试检查阶段**
   - 检查当前代理解析是否失败
   - 如果解析失败且可重试，返回相同代理进行重试

3. **决策执行阶段**
   - 根据workflow配置获取有效的下一步代理选项
   - 特殊处理verifier结果（成功则finish，失败则排除finish选项）
   - 根据选项数量和mandatory_analysis配置决定是否需要LLM分析

4. **智能决策**
   - 无选项：直接结束（finish）
   - 单选项且非强制分析：直接执行
   - 单选项且强制分析 或 多选项：调用LLM进行智能决策

## 关键方法说明

### record_agent_execution() - 代理执行记录
- **功能**：记录代理执行结果，进行解析并更新任务信息
- **流程**：保存原始数据到trace → 使用相应解析器解析结果 → 更新task_info
- **参数**：代理名称、执行结果、提示词、推理过程、错误日志、性能结果
- **返回**：解析是否成功

### _llm_decide_next_agent() - LLM智能决策
- **功能**：使用LLM分析当前状态并决策下一个代理
- **流程**：构建输入数据 → 调用LLM → 解析决策结果 → 保存建议
- **模板**：使用conductor/analyze.j2模板进行分析
- **输入**：当前代理、代理结果、错误日志、有效选项等
- **输出**：决策的下一个代理名称

### set_task_info() - 任务信息初始化
- **功能**：基于workflow配置初始化任务信息和基础文档
- **支持**：动态字段初始化、基础文档集成

## 用户自定义扩展

### 扩展概述
**Conductor模块作为智能调度中心**，基于workflow.yaml配置文件和LLM智能分析来调控任务走向。用户可以通过以下方式自定义扩展：

1. **配置文件扩展**：修改或创建新的workflow.yaml配置文件，定义自定义的代理流程
2. **代码扩展**：修改`get_next_agent()`函数，添加自定义的决策逻辑
3. **模板扩展**：修改conductor/analyze.j2模板，定制LLM分析的提示词

### 配置文件扩展示例
用户可以创建自定义的workflow配置，例如添加新的代理或修改流程：

```yaml
agent_info:
  designer:
    possible_next_agent: [coder, designer]  # 支持designer自我修复
  coder:
    possible_next_agent: [verifier, coder]
  verifier:
    possible_next_agent: [finish, designer, coder, optimizer]  # 添加optimizer代理
  optimizer:
    possible_next_agent: [verifier, optimizer]
start_agent: designer
mandatory_analysis: [verifier, optimizer]  # 强制对关键代理进行LLM分析
```

### 典型执行流程
基于默认配置的典型流程：

- `designer` → `conductor (决策)` → `coder`
- `coder` → `conductor (决策)` → `verifier`  
- `verifier` → `conductor (智能分析)` → `finish` / `coder`

通过Conductor的智能调度，各个代理形成**自适应的执行循环**，根据任务状态动态选择最优路径。

### 扩展要点
- **主要扩展入口**：`get_next_agent()`函数和workflow.yaml配置文件
- **决策逻辑**：可在`_llm_decide_next_agent()`中自定义LLM决策流程
- **状态管理**：通过`task_info`和`trace`访问完整的执行状态
- **解析扩展**：通过`record_agent_execution()`支持新代理的结果解析
