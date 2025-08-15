# Designer设计文档

## 概述
Designer是AI Kernel Generator中的核心组件，基于大语言模型(LLM)自动生成算法设计文档。它继承自`AgentBase`，负责根据算子名称、任务描述和硬件配置智能生成高质量的算法草图。Designer使用自定义设计语言来表达算法逻辑。

## 核心功能
- **智能设计生成**: 根据算子需求自动生成算法设计文档
- **文档驱动式接入集成（Doc-Driven Integration）**: 支持自定义参考文档以提升生成质量
- **多DSL支持**: 支持不同的设计语言
- **硬件感知设计**: 在设计生成过程中考虑硬件特性
- **文档集成**: 自动加载设计规范和参考资料

## 初始化参数
| 参数名称 | 类型/必需性 | 描述 |
|---------|---------|---------|
| op_name | str (必需) | 算子名称，标识特定算子 |
| task_desc | str (必需) | 任务描述，详细说明算子功能需求 |
| dsl | str (必需) | 设计语言："triton"、"swft"等 |
| backend | str (必需) | 硬件后端："ascend"、"cuda"等 |
| arch | str (必需) | 硬件架构："ascend910b4"、"a100"等 |
| workflow_config_path | str (可选) | 工作流配置文件路径（通常由 Task 根据编排配置注入） |
| config | dict (必需) | 完整编排配置，包含 log_dir、agent_model_config、docs_dir 等（详见《[任务编排方案配置](./TaskOrchestrationPlan.md)》） |

> 相关文档：工作流见《[Workflow](./Workflow.md)》；文档接入见《[文档驱动式接入指南](./DocDrivenIntegration.md)》。

## 文档驱动式接入集成（Doc-Driven Integration）

Designer 通过编排配置中的 `docs_dir.designer` 加载参考文档；文档清单与规范请见《[文档驱动式接入指南](./DocDrivenIntegration.md)》，此处不再重复。

## 执行流程

1. **初始化阶段**
   - 加载工作流配置并创建解析器
   - 初始化设计生成模板
   - 使用文档驱动式接入加载参考文档
   - 准备基础文档结构

2. **生成阶段**
   - 处理任务信息和conductor建议
   - 使用加载的文档执行LLM生成
   - 返回生成的设计、提示词和推理过程

3. **文档结构**
   - DSL规范和语法规则
   - 算法设计模式和示例
   - 硬件特定的考虑因素
   - 输出解析的格式指令
