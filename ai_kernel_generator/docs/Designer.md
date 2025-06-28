# 通用 Designer 设计文档

## 概述
Designer 是 AI Kernel Generator 中的核心组件，基于大型语言模型(LLM)自动生成和修复设计文档。它继承自 `AgentBase`，负责根据算子名称、任务描述和硬件配置，智能生成高质量的算子设计文档。当前我们采用AUL (AI Unity Language，算法统一语言) 作为设计文档的表达语言，用户可以灵活的设计其他实现方式。

## 核心功能
- **智能代码生成**：根据算子名称和任务描述自动生成 AUL 代码
- **代码自动修复**：基于验证反馈智能修复代码问题  
- **多硬件支持**：支持 Ascend NPU、CUDA GPU 等硬件后端
- **文档集成**：自动加载 AUL 规范和硬件文档
- **动态适配**：根据硬件类型动态获取配置信息

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称，如 "matmul", "relu" |
| task_desc | str (必选) | 任务描述，详细说明算子功能需求 |
| model_config | dict (必选) | LLM 模型配置，包含生成和修复两个模型配置 |
| impl_type | str (可选) | 实现类型，如 "swft"，默认："" |
| backend | str (可选) | 硬件后端：cpu/ascend/cuda，默认："" |
| arch | str (可选) | 硬件架构：ascend310p3/ascend910b4/a100，默认："" |

## 执行流程 run

1. **状态更新阶段**
   - 提取现有AUL代码（来自parsed_code.aul_code）
   - 调用update()更新代理状态信息

2. **核心执行阶段**  
   - 基于action_type选择对应的处理逻辑
     - `DO_DESIGNER`：使用aul_gen_prompt模板和aul_gen_input数据调用run_llm()
     - `FIX_DESIGNER`：使用aul_fix_prompt模板和aul_fix_input数据调用run_llm()
   - 不支持的action_type抛出ValueError异常

3. **结果返回**
   - 返回三元组：(生成内容, 格式化提示词, 推理内容)

## 使用示例
```python
from ai_kernel_generator.core.agent.aul_designer import AULDesigner
from ai_kernel_generator.core.utils import ActionType

# 创建Designer实例
designer = AULDesigner(
    op_name="relu",
    task_desc=task_desc,
    model_config=config["model"],
    impl_type="triton",
    backend="ascend",
    arch="ascend910b4"
)

# 执行代码生成
async def generate_code():
    result = await designer.run(
        action_type=ActionType.DO_DESIGNER,
        parsed_code=None,
        suggestions=""
    )
    print(f"生成的AUL代码: {result[0]}")

# 执行代码修复
async def fix_code():
    result = await designer.run(
        action_type=ActionType.FIX_DESIGNER,
        parsed_code=parsed_code,
        suggestions="优化内存访问模式"
    )
    print(f"修复后的代码: {result[0]}")
```
