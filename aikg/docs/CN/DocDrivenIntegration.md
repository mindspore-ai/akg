# 文档驱动式接入指南（Doc-Driven Integration, DDI）

## 引言

文档驱动式接入（DDI）以“文档即契约”为核心，通过统一的文档规范（DocSpec）与配置声明，使新的 DSL、前端框架与后端硬件在不修改 AIKG 本体的前提下完成接入，降低耦合与维护成本。

## 背景与问题分析

- 多样化生态（DSL/前端/后端）接入成本高，常需在框架内部进行定制化改造。
- 组件边界不清、耦合度高，升级或替换实现时牵连面广、风险大。
- 规范与示例分散，缺少统一约定，知识难以复用与沉淀。

## 解决方案概述

- 文档即契约（DocSpec）：以统一的文档结构、命名与内容约定描述能力边界。
- 角色解耦：`Designer`、`Coder` 等 Agent 基于文档契约工作，无需改动 AIKG 本体。
- 配置接入：在配置中声明文档目录，即可完成对接与替换。

## 文档接入“任务编排方案配置”

通过“任务编排方案配置”的 `docs_dir` 字段启用与配置。示例：`aikg/python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml`：

```yaml
# Documentation directory configuration
docs_dir:
  designer: "path/to/designer/docs"  # Designer参考文档目录
  coder: "path/to/coder/docs"        # Coder参考文档目录
```


## 文档要求（DocSpec）


### Agent 必需文档
- `basic_docs.md`：DSL 基础文档（核心概念、语法/语义、编译/执行模型、常见约束）。

### Agent 可选文档/代码
可选文档/代码，用于进一步优化 Agent 表现，推荐提供。
- `api/api.md`：目标框架/后端的 API 接口说明与注意事项。
- `suggestion_docs.md`：专家建议文档（性能优化提示、易错点等）。
- `examples/` 目录：建议以框架名称开头命名示例文件（如 `mindspore_*.py`）。

## 使用方法

1) 准备文档目录结构：

```
your_docs_dir/
├── basic_docs.md
├── api/
│   └── api.md  
├── suggestion_docs.md
└── examples/
    ├── mindspore_example_001.py
    ├── torch_example_001.py
    └── ...
```

2) 在配置文件中指定路径：

```yaml
docs_dir:
  designer: "path/to/your_docs_dir"
  coder: "path/to/your_docs_dir"
```

3) 应用配置：

```python
from ai_kernel_generator.config.config_validator import load_config

config = load_config(config_path="./your_custom_plan.yaml")
task = Task(op_name="custom_op", task_desc="...", config=config)
```

通过文档驱动式接入（DDI），只需按上述 DocSpec 要求准备文档并完成配置声明，即可在 AIKG 中高效接入新的 DSL、前端与后端。