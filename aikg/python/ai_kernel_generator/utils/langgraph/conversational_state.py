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

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from operator import add


# 对话历史最大长度配置
# 默认保留最近 40 条消息（约 20 轮对话）
# 可以根据实际需求调整：
# - 单个简单算子：20-40 条（10-20 轮）
# - 单个复杂算子：40-60 条（20-30 轮）
# - 多个算子场景：60-100 条（30-50 轮）
MAX_CONVERSATION_HISTORY_LENGTH = 40


def limit_conversation_history(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """
    限制对话历史长度的累积函数
    """
    # 合并现有历史和新消息
    combined = (existing or []) + (new or [])
    
    # 如果超过最大长度，只保留最近的消息
    if len(combined) > MAX_CONVERSATION_HISTORY_LENGTH:
        return combined[-MAX_CONVERSATION_HISTORY_LENGTH:]
    
    return combined


class Message(TypedDict):
    """对话消息"""
    role: str  # 'user' or 'assistant' or 'system'
    content: str
    timestamp: Optional[str]


class ConversationalOpGenState(TypedDict, total=False):
    """对话式算子生成的状态定义"""
    
    user_request: str  # 初始用户请求（用于 OpTaskBuilder）
    current_user_input: Optional[str]  # 当前轮的用户输入（用于多轮对话中判断task_type等）
    user_feedback: Optional[str]  # 用户对 task 代码的反馈
    user_confirmed: bool  # 用户是否确认 task 代码
    retry_requested: Optional[bool]  # 用户是否请求重新生成 task（在子 Agent 执行后）
    retry_sub_agent_only: Optional[bool]  # 用户是否只重新调用子 Agent（不重新生成 task）
    
    # === 子Agent选择相关 ===
    sub_workflow_specified_by_user: Optional[bool]  # 用户是否明确指定了子Agent
    
    # === 意图分类相关 ===
    last_intent: Optional[str]  # 上次的意图类型（operator_dev/general_question/unclear）
    last_intent_confidence: Optional[float]  # 上次的意图置信度（0.0-1.0）
    last_action_reasoning: Optional[str]  # 上次的动作分析推理（用于判断无关问题等）

    # 对话历史（自动限制长度，避免无限增长）
    conversation_history: Annotated[List[Message], limit_conversation_history]
    
    # === OpTaskBuildAgent 输出 ===
    task_code: Optional[str]  # 生成的 task 代码
    op_name: Optional[str]  # 算子名称
    op_description: Optional[str]  # 算子描述
    task_reasoning: Optional[str]
    task_init_status: Optional[str]  # OpTaskBuilder 的状态（ready/need_clarification/unsupported）  
    
    
    sub_workflow: Optional[str]  
    available_workflows: List[str]  
    
    generated_code: Optional[str]  
    generation_success: bool  
    generation_error: Optional[str]  
    
    
    verification_result: bool  
    verification_error: Optional[str]  
    profile_result: Optional[Dict[str, Any]]  
    
    
    iteration: int  
    max_iterations: int  
    current_step: str  
    should_continue: bool  
    

    framework: str  
    backend: str  
    arch: str 
    dsl: str  
    task_id: str  
    task_label: str  
    

    config: Dict[str, Any]  
    
    error_count: int  
    last_error: Optional[str]
    
    # === 前端显示相关 ===
    display_message: Optional[str]  # 格式化后的显示消息（供前端直接打印）
    hint_message: Optional[str]  # 提示消息（告诉用户下一步可以做什么）  


class SubAgentState(TypedDict, total=False):
    
    op_name: str
    task_desc: str  
    task_id: str
    task_label: str
    dsl: str
    framework: str
    backend: str
    arch: str
    workflow: str  
    config: Dict[str, Any]
    

    success: bool
    generated_code: Optional[str]
    error_message: Optional[str]
    verifier_result: bool
    verifier_error: Optional[str]
    profile_res: Optional[Dict[str, Any]]
