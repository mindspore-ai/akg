# Trace 模块设计文档

## 概述
Trace模块负责完整记录AI Kernel生成过程中的大模型推理痕迹，实现Designer、Coder、Verifier等组件的操作追踪和保存功能。通过标准化的日志格式和存储机制，支持生成过程的可追溯性和问题诊断。


## 类方法说明
### `insert_designer_or_coder_record()` 和 `insert_tester_record()`

**功能**
- 记录大模型的推理过程和生成结果，并保存到指定目录。

**参数说明**
| 参数名 | 类型 | 说明 |
|-------|-----|-----|
| res | str | 大模型输出的原始响应（包含代码和注释） |
| prompt | str | 本次操作的完整prompt模板 |
| reasoning | str | 大模型的推理过程文本 |
| action_type | ActionType | 操作类型 |

**存储内容**
- result.txt：解析后的纯代码+注释文档
- prompt.txt：完整的prompt模板
- reasoning.txt：推理过程文本

### `insert_tester_record()`

**功能**
- 记录测试结果和性能分析数据。

**参数说明**
| 参数名 | 类型 | 说明 |
|-------|-----|-----|
| verify_res | str | 验证结果|
| verify_log | str | 详细的验证日志 |
| profile | str | 性能分析数据（JSON格式） |
| action_type | ActionType | 操作类型 |

**存储内容**
- error_log.txt：详细的验证日志


### 文件命名规则
`I{任务ID}_S{步骤序号:02d}_{算子名称}_{操作类型}_参数名.txt`

**示例**
```
I123_S03_exp_add_op_DO_DESIGNER_prompt.txt
I123_S04_exp_add_op_DO_CODER_result.txt
```

## 文件存储规范
### 目录结构
```
log_dir/
└── op_name/
    ├── I{task_id}_S{step}_{op_name}_{action_type}_prompt.txt     # 完整的prompt模板
    ├── I{task_id}_S{step}_{op_name}_{action_type}_reasoning.txt  # 推理过程文本
    ├── I{task_id}_S{step}_{op_name}_{action_type}_result.txt     # 解析后的纯代码+注释文档
    ├── I{task_id}_S{step}_{op_name}_{action_type}_error_log.txt  # 详细的验证日志
    └── conductor/
        ├── I{task_id}_S{step}_{op_name}_CheckDesigner_prompt.txt  # designer self_check prompt
        ├── I{task_id}_S{step}_{op_name}_CheckDesigner_result.txt  # designer self_check result
        ├── I{task_id}_S{step}_{op_name}_CheckCoder_prompt.txt     # coder self_check prompt
        ├── I{task_id}_S{step}_{op_name}_CheckCoder_result.txt     # coder self_check result
        └── I{task_id}_S{step}_{op_name}_AnalyzeError_prompt.txt   # 错误分析 prompt
        └── I{task_id}_S{step}_{op_name}_AnalyzeError_result.txt   # 错误分析 result
    └── I{task_id}_S{step}_verify/
        ├── {op_name}_{framework}.py  # 原始任务mindspore/torch/numpy实现
        ├── {op_name}_{impl_type}.py  # swft/triton实现
        └── verify_{op_name}.py       # 错误分析
```


## 使用示例
```python
trace = Trace(op_name="swish", task_id="123", log_dir="~/logs")

trace.insert_designer_or_coder_record(
    res=designer_response,
    prompt=prompt_template,
    reasoning=llm_reasoning,
    action_type=ActionType.DO_DESIGNER
)
```
