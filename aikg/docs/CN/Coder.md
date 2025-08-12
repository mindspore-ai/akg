# Coder设计文档

## 概述
Coder是AI Kernel Generator中的核心组件，负责将算法设计转换为具体的实现代码。它继承自`AgentBase`，负责将高级算法草图（由Designer生成）转换为特定硬件平台和框架的可执行算子代码。

## 核心功能
- **设计到代码转换**: 将算法设计转换为具体实现代码
- **CustomDocs集成**: 利用自定义参考文档进行精确代码生成
- **多DSL支持**: 支持各种目标语言（Triton、SWFT等）
- **框架适配**: 为不同前端框架（MindSpore、PyTorch等）生成代码
- **错误引导修复**: 基于验证反馈修复生成的代码
- **API集成**: 基于文档智能选择合适的API

## 初始化参数
| 参数名称 | 类型/必需性 | 描述 |
|---------|---------|---------|
| op_name | str (必需) | 算子名称，标识特定算子 |
| task_desc | str (必需) | 任务描述，详细说明算子功能需求 |
| dsl | str (必需) | 目标DSL："triton"、"swft"等 |
| framework | str (必需) | 前端框架："mindspore"、"torch"、"numpy"等 |
| backend | str (必需) | 硬件后端："ascend"、"cuda"等 |
| arch | str (必需) | 硬件架构："ascend910b4"、"a100"等 |
| workflow_config_path | str (可选) | 工作流配置文件路径 |
| config | dict (必需) | 完整配置，包括CustomDocs设置 |

## CustomDocs集成

Coder广泛使用CustomDocs功能加载全面的参考资料：

### 必需文档
根据代码分析，以下文档对Coder的正常运行至关重要：

- `basic_docs.md` - DSL基础文档和语法规范
- `api/api.md` - API接口文档，包含详细的函数描述
- `suggestion_docs.md` - 专家建议和最佳实践
- `examples/` 目录 - 框架特定的示例文件（如`mindspore_*.py`、`torch_*.py`）

### 文档加载过程
Coder从配置中`docs_dir.coder`路径加载文档：
```python
self.base_doc = {
    "api_docs": self.load_doc("api/api.md"),
    "dsl_basic_docs": self.load_doc("basic_docs.md"),
    "dsl_examples": self._load_dsl_examples(),
    "expert_suggestion": self.load_doc("suggestion_docs.md"),
    # ... 其他字段
}
```

### 智能文档处理
- **API压缩**: 对于大型API文档，使用LLM提取相关API
- **示例加载**: 从examples目录动态加载框架特定的示例

## 配置示例
```yaml
# 在DSL配置文件中
docs_dir:
  coder: "resources/docs/triton_docs"  # Coder参考文档
  
agent_model_config:
  coder: "your_model_name"
  api_generator: "your_api_model"  # 用于API文档压缩
```

## 文档目录结构
```
your_coder_docs/
├── basic_docs.md           # DSL语法和基本概念
├── api/
│   └── api.md             # 全面的API文档
├── suggestion_docs.md     # 专家建议和模式
└── examples/
    ├── mindspore_example.py    # MindSpore特定示例
    ├── torch_example.py        # PyTorch特定示例
    ├── mindspore_matmul.py     # 操作特定示例
    └── ...
```

## 执行流程

1. **初始化阶段**
   - 加载工作流配置并创建解析器
   - 初始化代码生成模板
   - 使用CustomDocs加载所有参考文档
   - 准备包含加载内容的基础文档结构

2. **处理阶段**
   - 从任务信息中提取算法设计
   - 处理API文档（如需要则压缩）
   - 加载框架特定的示例
   - 整合conductor建议和错误反馈

3. **生成阶段**
   - 使用全面的上下文执行LLM生成
   - 返回生成的代码、提示词和推理过程

## 特殊功能

### 框架特定示例加载
Coder自动加载与指定框架匹配的示例：
```python
def _load_dsl_examples(self) -> str:
    # 从examples目录加载"mindspore_*.py"、"torch_*.py"等文件
    # 支持多种文件格式：.py、.md、.txt
```
