# Coder设计文档

## 概述
Coder是AI Kernel Generator中的核心组件，负责将算法设计转换为具体的实现代码。它继承自`AgentBase`，负责将高级算法草图（由Designer生成）转换为特定硬件平台和框架的可执行算子代码。

## 核心功能
- **设计到代码转换**: 将算法设计转换为具体实现代码
- **文档驱动式接入集成（Doc-Driven Integration）**: 利用自定义参考文档进行精确代码生成
- **多DSL支持**: 支持各种目标语言（Triton、SWFT等）
- **框架适配**: 为不同前端框架（MindSpore、PyTorch等）生成代码
- **错误引导修复**: 基于验证反馈修复生成的代码
- **API集成**: 基于文档智能选择合适的API

## 初始化参数
| 参数名称 | 类型/必需性 | 描述 |
|---------|---------|---------|
| op_name | str (必需) | 算子名称，标识特定算子 |
| task_desc | str (必需) | 任务描述，详细说明算子功能需求 |
| dsl | str (必需) | 目标DSL："triton_cuda"、"triton_ascend"、"swft"等 |
| framework | str (必需) | 前端框架："mindspore"、"torch"、"numpy"等 |
| backend | str (必需) | 硬件后端："ascend"、"cuda"等 |
| arch | str (必需) | 硬件架构："ascend910b4"、"a100"等 |
| workflow_config_path | str (可选) | 工作流配置文件路径（通常由 Task 根据编排配置注入） |
| config | dict (必需) | 完整编排配置，包含 log_dir、agent_model_config、docs_dir 等（详见《[任务编排方案配置](./TaskOrchestrationPlan.md)》） |

> 相关文档：工作流见《[Workflow](./Workflow.md)》；文档接入见《[文档驱动式接入指南](./DocDrivenIntegration.md)》。

## 文档驱动式接入集成（Doc-Driven Integration）

Coder 通过编排配置中的 `docs_dir.coder` 加载参考文档；文档清单、目录结构与写作规范请见《[文档驱动式接入指南](./DocDrivenIntegration.md)》，此处不再重复。

## 配置示例
```yaml
# 在任务编排方案配置文件中
docs_dir:
  coder: "resources/docs/triton_ascend_docs"  # Coder参考文档
  
agent_model_config:
  coder: "your_model_name"
  api_generator: "your_api_model"  # 用于API文档压缩
```

 

## 执行流程

1. **初始化阶段**
   - 加载工作流配置并创建解析器
   - 初始化代码生成模板
   - 使用文档驱动式接入（Doc-Driven Integration）加载所有参考文档
   - 准备包含加载内容的基础文档结构

2. **处理阶段**
   - 从任务信息中提取算法设计
   - 处理API文档（如需要则压缩）
   - 加载框架特定的示例
   - 整合conductor建议和错误反馈

3. **生成阶段**
   - 使用全面的上下文执行LLM生成
   - 返回生成的代码、提示词和推理过程

 
