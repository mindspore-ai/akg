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

"""State definition for OpTaskBuilder multi-turn interaction workflow."""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from operator import add


class OpTaskBuilderState(TypedDict, total=False):
    """State definition for OpTaskBuilder workflow.
    
    This state is used for multi-turn interaction to convert user's natural language
    requirements into KernelBench format op_task_desc.
    """
    
    # === 用户输入 ===
    user_input: str                    # 用户的初始文字需求
    user_feedback: Optional[str]       # 用户对生成结果的反馈/补充信息
    
    # === 多轮交互历史（累积）===
    conversation_history: Annotated[List[Dict[str, str]], add]  # 对话历史记录
    
    # === Agent输出 ===
    generated_task_desc: Optional[str]      # 生成的KernelBench格式代码
    clarification_question: Optional[str]   # 需要向用户澄清的问题
    modification_suggestion: Optional[str]  # 修改建议
    agent_reasoning: Optional[str]          # Agent的推理过程
    agent_message: Optional[str]            # 给用户的消息
    
    # === 任务配置（从用户需求中提取）===
    op_name: Optional[str]              # 算子名称
    framework: str                      # 框架，默认torch
    backend: str                        # 后端：cuda/ascend
    arch: str                           # 架构：a100/ascend910b4等
    dsl: str                            # DSL类型
    task_label: Optional[str]           # 任务标签（用于路由）
    
    # === 状态控制 ===
    status: str  # "need_clarification" | "need_modification" | "ready" | "unsupported"
    iteration: int                      # 当前交互轮次
    max_iterations: int                 # 最大交互轮次
    session_id: str                     # Session ID，用于 hook 管理（必填）
    
    # === 检查结果 ===
    static_check_passed: bool           # 静态检查是否通过
    static_check_error: Optional[str]   # 静态检查错误信息
    runtime_check_passed: bool          # 运行时检查是否通过
    runtime_check_error: Optional[str]   # 运行时检查错误信息
    check_retry_count: int              # 当前连续检查失败次数（内部重试计数，包括静态和运行时检查）
    max_check_retries: int              # 最大检查重试次数（从配置读取，包括静态和运行时检查）
    
    # === Prompt/日志记录 ===
    op_task_builder_prompt: Optional[str]     # OpTaskBuilder Agent使用的prompt


# Status常量定义
class OpTaskBuilderStatus:
    """OpTaskBuilder状态枚举"""
    READY = "ready"                     # 格式检查通过，可以启动原workflow
    NEED_CLARIFICATION = "need_clarification"  # 需要用户澄清
    NEED_MODIFICATION = "need_modification"    # 需要用户确认修改
    UNSUPPORTED = "unsupported"         # 不支持的需求
