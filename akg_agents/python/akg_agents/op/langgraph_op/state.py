# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""算子生成工作流状态定义

继承通用 BaseState，添加算子生成场景的专用字段。
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from operator import add
from akg_agents.core_v2.langgraph_base.base_state import BaseState


class KernelGenState(BaseState, total=False):
    """算子生成工作流状态
    
    继承通用状态 BaseState，添加算子生成场景的专用字段：
    - 算子基础信息（op_name, task_desc, dsl, framework, backend, arch）
    - Agent 输出（designer_*, coder_*, verifier_*, conductor_*）
    - 验证结果（verifier_result, profile_res, multi_case_error）
    - 额外配置（inspirations, meta_prompts, handwrite_suggestions）
    """
    
    # === 算子基础信息 ===
    op_name: str                    # 算子名称
    task_desc: str                  # 任务描述（框架代码）
    dsl: str                        # DSL 类型：triton_cuda, triton_ascend, swft
    framework: str                  # 框架：torch, mindspore, numpy
    backend: str                    # 后端：cuda, ascend
    arch: str                       # 架构：a100, ascend910b4
    task_type: str                  # 任务类型：profile, precision_only
    
    # === Designer 输出 ===
    designer_code: Optional[str]
    designer_prompt: Optional[str]
    designer_reasoning: Optional[str]
    
    # === Coder 输出 ===
    coder_code: Optional[str]
    coder_prompt: Optional[str]
    coder_reasoning: Optional[str]
    space_config_code: Optional[str]  # 动态形状参数空间
    
    # === Verifier 输出 ===
    verifier_result: bool
    verifier_error: str
    profile_res: Optional[Dict[str, Any]]
    
    # === 多 case 验证 ===
    multi_case_error: Optional[str]
    
    # === 历史记录（算子专用格式，累积）===
    history_attempts: Annotated[List[Dict[str, Any]], add]
    
    # === Code Checker 结果 ===
    code_check_passed: Optional[bool]
    code_check_errors: Optional[str]  # 格式化的错误信息
    code_check_details: Optional[List[Dict[str, Any]]]  # 详细错误列表

    # === 代码生成异常（max_tokens 截断等）===
    codegen_invalid: Optional[bool]
    codegen_invalid_reason: Optional[str]
    
    # === Conductor 建议 ===
    conductor_suggestion: Optional[str]
    conductor_decision: Optional[str]  # Conductor 的决策结果
    expert_suggestion: Optional[str]   # 专家建议文档（suggestion_docs.md 内容）
    conductor_step_count: Optional[int]

    # === 额外配置 ===
    inspirations: Optional[List[str]]
    meta_prompts: Optional[str]
    handwrite_suggestions: Optional[List[Dict[str, str]]]
    user_requirements: Optional[str]  # 用户额外需求（来自 ReAct 多轮对话）
    previous_code: Optional[str]      # 之前生成的代码（用于修改场景，来自 ReAct 多轮对话）
    
    # === Base Doc 字段 ===
    dsl_api_doc: Optional[str]
    framework_api_doc: Optional[str]
    backend_api_doc: Optional[str]
    workflow_name: Optional[str]
    
    # === 路径配置 ===
    cur_path: Optional[str]  # 自定义工作路径，用于控制中间文件和代码输出的存放位置
