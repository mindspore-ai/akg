# 自定义文档配置指南

## 概述

CustomDocs功能允许用户为AIKG中的不同Agent配置自定义参考文档，从而提升代码生成的质量和精度。通过为Agent提供相关的技术文档、API规范或算法说明，可以让AI更好地理解上下文并生成更准确的代码。

## 配置方式

CustomDocs功能通过DSLConfig配置系统实现，主要在配置文件的`docs_dir`部分进行设置：

### 基本配置结构

```yaml
docs_dir:
  designer: "path/to/designer/docs"  # Designer参考文档目录
  coder: "path/to/coder/docs"        # Coder参考文档目录
```

## 必需文档类型

根据代码分析，以下文档是各Agent正常工作所必需的：

### Designer必需文档
- `basic_docs.md` - DSL基础文档

### Coder必需文档
- `basic_docs.md` - DSL基础文档  
- `api/api.md` - API接口文档
- `suggestion_docs.md` - 专家建议文档
- `examples/` 目录 - 包含以framework名称开头的示例文件（如`mindspore_*.py`）

## 使用方法

1. **准备文档目录结构**：
```
your_docs_dir/
├── basic_docs.md
├── api/
│   └── api.md  
├── suggestion_docs.md
└── examples/
    ├── mindspore_example.py
    ├── torch_example.py
    └── ...
```

2. **在配置文件中指定路径**：
```yaml
docs_dir:
  designer: "path/to/your_docs_dir"
  coder: "path/to/your_docs_dir"
```

3. **应用配置**：
```python
from ai_kernel_generator import load_config

config = load_config("your_custom_config")
task = Task(op_name="custom_op", task_desc="...", config=config)
```

通过CustomDocs功能，用户可以为不同的DSL和使用场景定制最适合的参考文档，从而显著提升AIKG的代码生成效果。 