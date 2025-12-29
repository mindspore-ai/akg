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

"""
MainOpAgent 辅助工具函数
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)



def is_operator_related_intent(intent: str, confidence: float, threshold: float = 0.6) -> bool:
    """
    判断意图是否与算子开发相关
    
    Args:
        intent: 意图分类结果
        confidence: 置信度
        threshold: 置信度阈值
        
    Returns:
        是否与算子开发相关
    """
    if confidence < threshold:
        # 置信度太低，无法判断
        return True  # 默认允许继续（善意理解）
    
    operator_related = [
        "operator_development",
        "operator_dev",  # 添加简短形式
        "code_generation",
        "modification",
        "optimization",
        "unclear"
    ]
    
    # 更宽松的判断：只要包含 "operator" 关键词，就认为是相关的
    if "operator" in intent.lower():
        return True
    
    return intent in operator_related


def quick_match_sub_agent_preference(user_input: str) -> Optional[str]:
    """
    快速匹配用户是否明确指定了子 Agent 偏好
    
    Args:
        user_input: 用户输入
        
    Returns:
        子 Agent 名称 (codeonly/evolve/kernel_verifier) 或 None
    """
    user_input_lower = user_input.lower()
    
    # 1. 检查是否同时包含"生成"和"性能测试"
    has_generate = any(kw in user_input_lower for kw in [
        "生成", "创建", "写", "实现", "generate", "create", "implement", "write"
    ])
    
    has_performance_test = any(kw in user_input_lower for kw in [
        # 性能测试相关
        "性能测试", "测试性能", "性能分析", "分析性能",
        "验证性能", "验证一下性能", "验证下性能",
        "测试一下性能", "测下性能", "测一下性能",
        "性能怎么样", "加速比", "速度对比", "性能对比",
        "看看性能", "看下性能", "看一下性能",
        "跑一下性能", "跑下性能",
        # 带"并"的组合
        "并测试", "并进行测试", "并测试性能", "并进行性能测试",
        "并分析性能", "并验证性能", "并benchmark", "并测一下",
        # 英文
        "benchmark", "performance test", "test performance", "profile",
        "check performance", "verify performance",
        "and test", "and profile", "and benchmark"
    ])
    
    # 如果同时包含"生成"和"性能测试"，返回 None（让 codeonly 处理，task_type="profile"）
    if has_generate and has_performance_test:
        logger.info("User requests generate + profile, returning None (codeonly will handle with task_type='profile')")
        return None
    
    # 2. 如果只有性能测试（没有生成），明确返回 kernel_verifier
    if has_performance_test and not has_generate:
        logger.info("User requests pure performance test, returning 'kernel_verifier'")
        return "kernel_verifier"
    
    # 3. 检查是否明确要求使用 evolve（必须是纯优化相关，排除性能测试）
    # 注意：这些关键词应该明确表示"要求最高性能/多轮优化"，而不是"测试性能"
    evolve_keywords = [
        # 明确指定evolve
        "使用evolve", "用evolve", "换evolve", "改用evolve",
        "use evolve", "evolve", "进化",
        # 多轮优化相关
        "多轮优化", "迭代优化", "自动调优", "进化优化",
        # 极致性能要求（与"性能测试"区分）
        "极致性能", "最优性能", "最佳性能", 
        "性能最优", "性能极致",
        "高性能算子", "超高性能", 
        # 明确的优化需求
        "追求性能", "优化到极致", "最大化性能"
    ]
    
    if any(kw in user_input_lower for kw in evolve_keywords):
        logger.info("User explicitly requests evolve (high performance optimization)")
        return "evolve"
    
    # 4. 检查是否明确要求使用 codeonly
    codeonly_keywords = [
        "使用codeonly", "用codeonly", "换codeonly", "改用codeonly",
        "use codeonly", "快速生成", "直接生成"
    ]
    
    if any(kw in user_input_lower for kw in codeonly_keywords):
        logger.info("User explicitly requests codeonly")
        return "codeonly"
    
    # 默认返回 None，让 LLM 进行选择
    return None


def extract_sub_agent_from_reasoning(reasoning: str) -> Optional[str]:
    """
    从 LLM 的推理过程中提取子 Agent 名称
    
    Args:
        reasoning: LLM 的推理过程
        
    Returns:
        子 Agent 名称 或 None
    """
    reasoning_lower = reasoning.lower()
    
    if "evolve" in reasoning_lower:
        return "evolve"
    elif "codeonly" in reasoning_lower:
        return "codeonly"
    elif "kernel_verifier" in reasoning_lower or "性能" in reasoning_lower:
        return "kernel_verifier"
    
    return None


def user_explicitly_requests_evolve(user_request: str, conversation_history: list) -> bool:
    """
    判断用户是否明确要求使用 evolve
    
    Args:
        user_request: 用户请求
        conversation_history: 对话历史
        
    Returns:
        是否明确要求 evolve
    """
    user_request_lower = user_request.lower()
    
    # 明确的 evolve 关键词（不包含性能相关）
    evolve_keywords = [
        "使用evolve", "用evolve", "换evolve", "改用evolve",
        "use evolve", "with evolve",
        "多轮优化", "迭代优化", "进化优化", "自动调优",
        "多次迭代", "iterative", "multi-round"
    ]
    
    if any(kw in user_request_lower for kw in evolve_keywords):
        logger.info(f"User explicitly requests evolve: matched keyword")
        return True
    
    return False


def user_requests_profile(user_request: str) -> bool:
    """判断用户是否要求性能测试（用于 codeonly task_type 判断）"""
    if not user_request:
        return False
    
    user_request_lower = user_request.lower()
    
    # 关键判断：同时包含"生成"和"性能测试"相关词汇
    has_generate = any(kw in user_request_lower for kw in [
        "生成", "创建", "写", "实现", "generate", "create", "implement", "write"
    ])
    
    has_performance_test = any(kw in user_request_lower for kw in [
        # 性能测试相关
        "性能测试", "测试性能", "性能分析", "分析性能",
        "验证性能", "验证一下性能", "验证下性能",
        "测试一下性能", "测下性能", "测一下性能",
        "性能怎么样", "加速比", "速度对比", "性能对比",
        "看看性能", "看下性能", "看一下性能",
        "跑一下性能", "跑下性能",
        "并测试", "并进行测试", "并测试性能", "并进行性能测试",
        "并分析性能", "并验证性能", "并benchmark", "并测一下",
        "benchmark", "performance test", "test performance", "profile",
        "check performance", "verify performance",
        "and test", "and profile", "and benchmark"
    ])
    
    # 如果同时包含"生成"和"性能测试"，返回True
    if has_generate and has_performance_test:
        logger.info(f"User requests generate + profile (has_generate=True, has_performance_test=True)")
        return True
    
    return False


def is_modification_request(user_input: str, has_previous_code: bool) -> bool:
    """
    判断用户输入是否是对现有代码的修改请求
    
    Args:
        user_input: 用户输入
        has_previous_code: 是否已有生成的代码
        
    Returns:
        是否是修改请求
    """
    if not has_previous_code:
        return False
    
    user_input_lower = user_input.lower()
    
    # 修改相关的关键词
    modification_keywords = [
        "修改", "改成", "改为", "换成", "调整",
        "shape", "dtype", "batch", "size", "dim",
        "modify", "change", "update", "adjust"
    ]
    
    # 检查是否是短输入 + 包含修改关键词
    is_short = len(user_input) < 100
    has_modification_keyword = any(kw in user_input_lower for kw in modification_keywords)
    
    if is_short and has_modification_keyword:
        logger.info(f"Detected potential modification request (short input with modification keyword)")
        return True
    
    return False


def simple_action_heuristic(state: dict, user_input: str) -> tuple:
    """
    简单的启发式规则判断用户动作（LLM 分析失败时的回退方案）
    
    Args:
        state: 当前状态
        user_input: 用户输入
        
    Returns:
        tuple: (action, is_new_operator, is_irrelevant, has_provided_task_code, is_complete_code, extracted_task_code, extracted_op_name, extracted_op_description)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    user_input_lower = user_input.lower()
    user_input_stripped = user_input.strip()
    
    # 检测明显的无关问题（即使 LLM 失败也应该拒绝）
    # 1. 身份询问
    identity_keywords = ["你是谁", "who are you", "你叫什么", "what's your name", "你能做什么", "what can you do"]
    if any(kw in user_input_lower for kw in identity_keywords):
        logger.info(f"Heuristic: Detected identity question, marking as irrelevant")
        return "cancel", False, True, False, False, '', '', ''
    
    # 2. 闲聊话题
    chitchat_keywords = ["天气", "weather", "笑话", "joke", "故事", "story", "新闻", "news"]
    if any(kw in user_input_lower for kw in chitchat_keywords):
        logger.info(f"Heuristic: Detected chitchat topic, marking as irrelevant")
        return "cancel", False, True, False, False, '', '', ''
    
    # 3. 完全无关的问题（输入太短且不包含算子相关词汇）
    if len(user_input_stripped) < 20:
        operator_related_keywords = [
            "算子", "operator", "kernel", "生成", "generate", "实现", "implement",
            "relu", "sigmoid", "tanh", "softmax", "matmul", "conv", "layernorm",
            "torch", "triton", "cuda", "代码", "code", "性能", "performance"
        ]
        has_operator_keyword = any(kw in user_input_lower for kw in operator_related_keywords)
        if not has_operator_keyword and not state.get("task_code"):
            # 输入很短，没有算子关键词，且对话刚开始 → 可能是无关问题
            logger.info(f"Heuristic: Short input with no operator keywords, marking as irrelevant")
            return "cancel", False, True, False, False, '', '', ''
    
    # 确认关键词
    confirm_keywords = ["确认", "ok", "yes", "好的", "可以", "继续", "confirm"]
    if any(kw in user_input_lower for kw in confirm_keywords):
        return "confirm", False, False, False, False, '', '', ''
    
    # 重试关键词
    retry_keywords = ["重新", "再试", "retry", "重做"]
    if any(kw in user_input_lower for kw in retry_keywords):
        if state.get("generated_code"):
            return "retry_sub_agent", False, False, False, False, '', '', ''
        else:
            return "retry", False, False, False, False, '', '', ''
    
    # 默认：修改（假设相关）
    return "revise", False, False, False, False, '', '', ''


def format_agents_info_for_llm(agents_info: dict) -> str:
    """
    格式化子 Agent 信息供 LLM 选择
    
    Args:
        agents_info: 子 Agent 详细信息字典
        
    Returns:
        格式化的字符串
    """
    formatted = []
    for name, info in agents_info.items():
        formatted.append(f"### {name}")
        formatted.append(f"- 描述: {info.get('description', '')}")
        formatted.append(f"- 适用场景: {info.get('use_cases', '')}")
        formatted.append(f"- 优势: {info.get('advantages', '')}")
        formatted.append("")
    
    return "\n".join(formatted)
