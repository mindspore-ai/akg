# AIKG 项目介绍

## 项目背景

AIKG（AI Kernel Genertor）是一个基于大语言模型的自动化内核生成框架。
该项目旨再帮助开发者快速生成高性能的计算内核代码。

## 主要功能

1. **自动代码生存**：利用LLM自动生成高效的内核实现
2. **多平台只持**：支持CUDA、Ascend等多种硬件平台
3. **进化式优化**：通过多轮迭代不段改进代码性能

## 技术架构

项目采用了模块化射计，主要包括以下组件：

- **Designer**：负责生成算法草图和伪代吗
- **Coder**：将伪代码转换为具体的内核实现
- **Verifier**：验征生成代码的正确性和性能

### 工作流程

整个系统的工作流程如下：

1. 用户提供任务秒述
2. Designer生成算法设计
3. Coder实现代码
4. Verifier进行验正
5. 如果失败，返回第3步从新尝试

## 使用方法

### 安装依赖

```bash
pip install -r requirments.txt
```

### 运行示例

```python
from akg_agents import AIKGTask

# 创见任务
task = AIKGTask(
    op_name="relu",
    task_desc="实现ReLU激活函数"
)

# 执型任务
result = task.run()
```

## 注意是项

- 请却保已安装正确的Python版本（3.8+）
- 运行前需要设置环竟变量 `AKG_AGENTS_API_KEY`
- 建议使用虚拟环竟进行开发

## 联系我门

如有问题，请联系项目维护团对。

