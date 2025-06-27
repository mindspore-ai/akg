# AUL_Designer说明

## 概述

AUL_Designer是AIKG框架中专门用于生成AUL（AI Unity Language）算子代码的设计器。它基于用户提供的算子任务描述，通过LLM+Prompt的方式自动生成高质量的AUL算子实现代码，并通过多阶段验证确保代码的正确性和性能。

AUL_Designer采用四阶段流水线设计：
1. **gen_sketch**: 生成AUL代码草图和tiling函数
2. **fix_aul_code**: 修复AUL代码中的语法和逻辑错误
3. **trans_aul_code_to_numpy_code**: 将AUL代码转换为等价的Numpy代码
4. **verify_numpy_from_aul_code**: 验证转换后的Numpy代码正确性

## 细节

### 核心流程

```
gen_sketch -> fix_aul_code -> trans_aul_code_to_numpy_code -> verify_numpy_from_aul_code
```

**gen_sketch阶段**：
- 基于算子任务描述和AUL规范生成初始AUL代码
- 同时生成对应的tiling函数实现
- 支持并行生成多个版本的代码样本

**fix_aul_code阶段**：
- 检查并修复AUL代码中的语法错误
- 优化tiling函数的实现逻辑
- 确保代码符合AUL规范要求

**trans_aul_code_to_numpy_code阶段**：
- 将修复后的AUL代码转换为等价的Numpy实现
- 保持算子的数学语义一致性
- 便于后续验证和测试

**verify_numpy_from_aul_code阶段**：
- 使用NumpyVerify验证转换后的代码
- 确保算子实现的正确性
- 过滤掉验证失败的代码版本

### 并行执行机制

AUL_Designer支持通过`run_parallel`机制并行执行多个任务：
- 使用`pack_tasks_with_mapping`和`unpack_results_with_mapping`函数管理任务映射
- 支持lambda并行方法提高生成效率
- 可配置`samples_num`参数生成多个代码版本

### 输入和目标

**输入参数**：
- `op_name`: 算子名称
- `op_task_str`: 算子任务描述（通常来自`*_op_task.py`文件）
- `samples_num`: 并行生成样本数量（默认为1）
- `preset_name`: LLM预设配置（如"deepseek_r1_default"）
- `parallel_method`: 并行执行方法（默认为"lambda"）
- `record_mode`: 是否记录推理过程（默认为False）

**输出目标**：
- 生成符合AUL规范的高质量算子代码
- 通过验证的可执行AUL实现
- 详细的调试信息和中间结果

### 提示模板系统

AUL_Designer使用以下核心提示模板：

**prompts模板**：
- `aul_template.j2`: AUL代码生成的主模板
- `aul_verify.j2`: AUL代码修复和验证模板
- `aul_to_numpy.j2`: AUL到Numpy转换模板
- `aul_tiling.j2`: Tiling函数生成指南

- 'aul_base.md': AUL规范基础定义
- 'aul_rules.md': AUL语法及执行方式
- 'aul_npu.md': NPU上的AUL拓展
- 'aul_npu_templetes.md': NPU-AUL样例代码
- 'aul_suggestions.md': 规则补充

**hardware information**：
- `Ascend310P3.j2`: Ascend310P3硬件信息和约束

### 记录和调试

**records输出**：
- `aul_origin`: 原始生成的AUL代码（`test_{op_name}_original_v{i}.py`）
- `aul_fixed`: 修复后的AUL代码（`test_{op_name}_fixed_v{i}.py`）
- `trans_numpy_code`: 转换后的Numpy代码（`test_{op_name}_numpy_v{i}.py`）
- `data`: 验证过程中的数据文件

调试信息保存在临时目录中，可通过日志级别控制是否保留。

## Designer新增场景

整体流程参考：`tests/ut/test_aul_designer.py`

### 新增op_task

在`tests/resources/`目录下新增task，如`XXX_op_task.py`，并在`tests/ut/test_aul_designer.py`中添加测试用例。

**XXX_op_task.py示例**
```python
import numpy as np
from save_data_utils import save_data_and_json


def xxx_op_impl(input_data):  # 算子的初始numpy实现
    # 实现具体的算子逻辑
    result = np.some_operation(input_data)
    return result


def xxx_op_host_run():  # 算子的host运行逻辑
    # 生成测试数据
    input_np = np.random.normal(0.0, 0.5, size=(40, 256)).astype(np.float16)
    expected = xxx_op_impl(input_np)
    
    # 保存测试数据和期望结果
    save_data_and_json(
        input_list=[input_np], 
        output_list=[], 
        expect_list=[expected], 
        op_name="xxx_op"
    )
    return expected

def xxx_op_host_run_both():
    input_np = np.random.normal(0.0, 0.5, size=(40, 256)).astype(np.float16)
    expected = xxx_op_impl(input_np)

    tiling = list()
    ouput_np = np.zeros_like(expected).astype(np.float16)
    add_op_impl_tiling(input_np, ouput_np, tiling)
    add_op_impl_npu(input_np, ouput_np, tiling)
    save_data_and_json(input_list=[input_np], output_list=[ouput_np], expect_list=[expected], op_name="add_op")
    return expected


if __name__ == "__main__":
    import sys
    mode = "expect_only"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "both":
        xxx_op_host_run_both()
    else:
        xxx_op_host_run()
```

**测试用例示例**
```python
@pytest.mark.level0
def test_xxx_op():
    op_name = "xxx_op"
    op_task_str = get_op_task_str(op_name)
    designer = AulDesigner(
        op_name=op_name, 
        op_task_str=op_task_str,
        preset_name="deepseek_r1_default",
        samples_num=1,
        supported_api="SWFT"
    )
    
    res, grouped_code = designer.run_with_self_check()
    assert res.success, "Test failed"
```

### 运行

参考`README.md`，运行以下命令测试特定算子：

```bash
# 测试单个算子
pytest -sv tests/ut/test_aul_designer.py::test_xxx_op

# 测试所有预定义算子
pytest -sv tests/ut/test_aul_designer.py::test_aul_designer

# 测试benchmark算子
pytest -sv tests/ut/test_aul_designer.py::test_aul_designer_from_benchmark
```

AUL_Designer会根据task尝试生成对应的AUL算子代码并自动校验正确性。如果结果错误，可以根据R1的thinking日志分析原因。

### 修改template

如果AUL_Designer生成的代码不符合预期，可以临时修改相应的模板文件：

**主要模板文件位置**：
- `ai_kernel_generator/resources/prompts/generation/aul/aul_template.j2` - AUL代码生成模板
- `ai_kernel_generator/resources/prompts/generation/aul/aul_verify.j2` - 代码修复模板
- `ai_kernel_generator/resources/prompts/generation/aul/aul_to_numpy.j2` - AUL到Numpy转换模板
- `ai_kernel_generator/resources/prompts/generation/aul/aul_tiling.j2` - Tiling函数指南

**模板修改建议**：
1. 根据具体算子的特点调整生成策略
2. 添加特定的错误处理逻辑
3. 优化tiling函数的生成规则
4. 增强AUL规范的约束检查

### 配置选项

**预设配置**：
- `deepseek_r1_default`: DeepSeek R1模型默认配置
- `volc_ds_r1_default`: 火山引擎DeepSeek R1配置
- `sflow_qwen_32b`: Qwen 32B模型配置
- ... 详见ai_kernel_generator/core/llm/llm_config.yaml

**并行方法**：
- `lambda`: Lambda函数包装并行执行
- `asyncio`: 异步执行
- `wrapper_invoke`: 分任务批量并行执行

**记录模式**：
- `record_mode=True`: 保存详细的推理过程到database目录
- `record_mode=False`: 仅保存最终结果