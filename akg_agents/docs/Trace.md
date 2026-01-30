# Trace Module Design Document

## Overview
The Trace module is responsible for completely recording the inference traces of the large model during the AI Kernel generation process. It implements the tracking and saving functions for operations of components like Designer, Coder, and Verifier. Through standardized log formats and storage mechanisms, it supports the traceability and problem diagnosis of the generation process.


## Class Method Descriptions
### `insert_designer_or_coder_record()` and `insert_verifier_record()`

**Function**
- Records the inference process and generation results of the large model and saves them to a specified directory.

**Parameter Description**
| Parameter Name | Type | Description |
|-------|-----|-----|
| res | str | The raw response from the large model (including code and comments). |
| prompt | str | The complete prompt template for this operation. |
| reasoning | str | The reasoning process text from the large model. |
| action_type | ActionType | The type of action. |

**Stored Content**
- result.txt: The parsed pure code + commented document.
- prompt.txt: The complete prompt template.
- reasoning.txt: The reasoning process text.

### `insert_verifier_record()`

**Function**
- Records test results and performance analysis data.

**Parameter Description**
| Parameter Name | Type | Description |
|-------|-----|-----|
| verify_res | str | The verification result.|
| verify_log | str | Detailed verification log. |
| profile | str | Performance analysis data (JSON format). |
| action_type | ActionType | The type of action. |

**Stored Content**
- error_log.txt: Detailed verification log.


### File Naming Convention
`I{TaskID}_S{StepNumber:02d}_{KernelName}_{ActionType}_{ParameterName}.txt`

**Example**
```
I123_S03_exp_add_op_DO_DESIGNER_prompt.txt
I123_S04_exp_add_op_DO_CODER_result.txt
```

## File Storage Specification
### Directory Structure
```
log_dir/
└── op_name/
    ├── I{task_id}_S{step}_{op_name}_{action_type}_prompt.txt     # Complete prompt template
    ├── I{task_id}_S{step}_{op_name}_{action_type}_reasoning.txt  # Reasoning process text
    ├── I{task_id}_S{step}_{op_name}_{action_type}_result.txt     # Parsed pure code + commented document
    ├── I{task_id}_S{step}_{op_name}_{action_type}_error_log.txt  # Detailed verification log
    └── conductor/
        ├── I{task_id}_S{step}_{op_name}_CheckDesigner_prompt.txt  # designer self_check prompt
        ├── I{task_id}_S{step}_{op_name}_CheckDesigner_result.txt  # designer self_check result
        ├── I{task_id}_S{step}_{op_name}_CheckCoder_prompt.txt     # coder self_check prompt
        ├── I{task_id}_S{step}_{op_name}_CheckCoder_result.txt     # coder self_check result
        └── I{task_id}_S{step}_{op_name}_AnalyzeError_prompt.txt   # Error analysis prompt
        └── I{task_id}_S{step}_{op_name}_AnalyzeError_result.txt   # Error analysis result
    └── I{task_id}_S{step}_verify/
        ├── {op_name}_{framework}.py  # Original task mindspore/torch/numpy implementation
        ├── {op_name}_{impl_type}.py  # swft/triton implementation
        └── verify_{op_name}.py       # Error analysis
```


## Usage Example
```python
trace = Trace(op_name="swish", task_id="123", log_dir="~/logs")

trace.insert_designer_or_coder_record(
    res=designer_response,
    prompt=prompt_template,
    reasoning=llm_reasoning,
    action_type=ActionType.DO_DESIGNER
)
``` 