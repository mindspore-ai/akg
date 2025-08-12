# Designer设计文档

## 概述
Designer是AI Kernel Generator中的核心组件，基于大语言模型(LLM)自动生成算法设计文档。它继承自`AgentBase`，负责根据算子名称、任务描述和硬件配置智能生成高质量的算法草图。Designer使用自定义设计语言来表达算法逻辑。

## 核心功能
- **智能设计生成**: 根据算子需求自动生成算法设计文档
- **CustomDocs集成**: 支持自定义参考文档以提升生成质量
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
| workflow_config_path | str (可选) | 工作流配置文件路径 |
| config | dict (必需) | 完整配置，包括CustomDocs设置 |

## CustomDocs集成

Designer利用CustomDocs功能从配置的目录中加载参考文档：

### 必需文档
- `basic_docs.md` - DSL基础文档和语法规范

### 文档加载
Designer从配置中`docs_dir.designer`路径加载文档：
```python
self.base_doc = {
    "dsl_basic_docs": self.load_doc("basic_docs.md"),
    # ... 其他字段
}
```

## 执行流程

1. **初始化阶段**
   - 加载工作流配置并创建解析器
   - 初始化设计生成模板
   - 使用CustomDocs加载参考文档
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
