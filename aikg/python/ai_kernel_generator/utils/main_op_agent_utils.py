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

提供意图判断、子Agent选择、修改请求检测等辅助功能
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def is_operator_related_intent(intent: str, confidence: float, threshold: float = 0.6) -> bool:
    """判断是否是算子相关的意图
    
    Args:
        intent: 意图类型（operator_dev/general_question/unclear）
        confidence: 置信度（0.0-1.0）
        threshold: 置信度阈值
        
    Returns:
        bool: 是否是算子相关
    """
    if intent == "operator_dev":
        # 明确是算子开发相关
        return True
    elif intent == "general_question" and confidence >= threshold:
        # 高置信度的一般问题 → 拒绝
        return False
    else:
        # unclear 或低置信度 → 让后续流程处理（安全策略）
        return True


def quick_match_sub_agent_preference(user_input: str) -> Optional[str]:
    """快速匹配用户是否明确要求使用特定的子Agent（字符串匹配，减少LLM调用）
    
    Args:
        user_input: 用户输入
        
    Returns:
        Optional[str]: 如果匹配到，返回子Agent名称 ("evolve"、"codeonly" 或 "kernel_verifier")，否则返回 None
    """
    if not user_input:
        return None
        
    user_input_lower = user_input.lower()
    
    # 关键判断：是否包含"生成"相关词汇
    has_generate = any(kw in user_input_lower for kw in [
        "生成", "创建", "写", "实现", "generate", "create", "implement", "write"
    ])
    
    # 关键判断：是否包含"性能测试"相关词汇
    has_performance_test = any(kw in user_input_lower for kw in [
        "性能测试", "测试性能", "性能分析", "分析性能",
        "验证性能", "性能怎么样", "加速比", "速度对比", "性能对比",
        "benchmark", "performance test", "test performance", "profile",
        "check performance", "verify performance",
        "测试", "测一下", "跑一下", "看看性能"
    ])
    
    # 场景1：包含"生成" + "性能测试" → 返回 None，让LLM判断或走默认codeonly（会设置task_type="profile"）
    if has_generate and has_performance_test:
        logger.info(f"User requests BOTH generate AND performance test, returning None (will use codeonly with task_type='profile')")
        return None  # 返回 None，让后续逻辑选择 codeonly，task_type会被设置为profile
    
    # 场景2：只包含"性能测试"，不包含"生成" → kernel_verifier
    if has_performance_test and not has_generate:
        logger.info(f"User requests ONLY performance test (no generate), returning kernel_verifier")
        return "kernel_verifier"
    
    # 检测换成 evolve 的关键词
    # 注意：只包含明确的 evolve 关键词和纯优化词，不包含"性能"相关词
    switch_to_evolve_keywords = [
        # 明确提到 evolve
        "换evolve", "用evolve", "使用evolve", "改用evolve", "试试evolve", "换成evolve",
        "换用evolve", "改为evolve", "切换evolve", "选evolve", "要evolve",
        "evolve重新生成", "evolve生成", "evolve优化", "用一下evolve", "走evolve",
        "switch to evolve", "use evolve", "try evolve", "with evolve", "change to evolve",
        # 纯优化关键词（不涉及性能）
        "多轮优化", "迭代优化", "进化优化", "自动调优", "多次迭代",
        "iterative optimization", "evolutionary optimization", "auto-tune"
    ]
    
    for keyword in switch_to_evolve_keywords:
        if keyword in user_input_lower:
            logger.info(f"Quick matched evolve keyword: '{keyword}'")
            return "evolve"
    
    # 检测换成 codeonly 的关键词
    switch_to_codeonly_keywords = [
        "换codeonly", "用codeonly", "使用codeonly", "改用codeonly", "试试codeonly", "换成codeonly",
        "换用codeonly", "改为codeonly", "切换codeonly", "选codeonly", "要codeonly",
        "codeonly重新生成", "codeonly生成", "用一下codeonly",
        "switch to codeonly", "use codeonly", "try codeonly", "with codeonly", "change to codeonly"
    ]
    
    for keyword in switch_to_codeonly_keywords:
        if keyword in user_input_lower:
            logger.info(f"Quick matched codeonly keyword: '{keyword}'")
            return "codeonly"
    
    return None


def extract_sub_agent_from_reasoning(reasoning: str) -> Optional[str]:
    """从LLM的推理中提取用户要求的子Agent
    
    Args:
        reasoning: LLM推理文本
        
    Returns:
        Optional[str]: 如果LLM推理中提到子Agent，返回名称，否则返回 None
    """
    if not reasoning:
        return None
    
    reasoning_lower = reasoning.lower()
    
    # 检查推理中是否提到 evolve
    evolve_indicators = [
        "使用 evolve", "用 evolve", "evolve 子agent", "evolve子agent",
        "要求使用 evolve", "明确要求 evolve", "指定 evolve",
        "use evolve", "using evolve", "evolve sub-agent", "evolve subagent"
    ]
    
    for indicator in evolve_indicators:
        if indicator in reasoning_lower:
            logger.info(f"LLM reasoning indicates evolve: '{indicator}'")
            return "evolve"
    
    # 检查推理中是否提到 codeonly
    codeonly_indicators = [
        "使用 codeonly", "用 codeonly", "codeonly 子agent", "codeonly子agent",
        "要求使用 codeonly", "明确要求 codeonly", "指定 codeonly",
        "use codeonly", "using codeonly", "codeonly sub-agent", "codeonly subagent"
    ]
    
    for indicator in codeonly_indicators:
        if indicator in reasoning_lower:
            logger.info(f"LLM reasoning indicates codeonly: '{indicator}'")
            return "codeonly"
    
    return None


def user_requests_profile(user_request: str) -> bool:
    """判断用户是否要求性能测试（用于 codeonly task_type 判断）
    
    这个函数用于判断当走codeonly时，是否需要设置task_type="profile"
    
    Args:
        user_request: 用户输入
        
    Returns:
        bool: 是否要求性能测试
    """
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
        # 带"并"的组合
        "并测试", "并进行测试", "并测试性能", "并进行性能测试",
        "并分析性能", "并验证性能", "并benchmark", "并测一下",
        # 英文
        "benchmark", "performance test", "test performance", "profile",
        "check performance", "verify performance",
        "and test", "and profile", "and benchmark"
    ])
    
    # 如果同时包含"生成"和"性能测试"，返回True
    if has_generate and has_performance_test:
        logger.info(f"User requests generate + profile (has_generate=True, has_performance_test=True)")
        return True
    
    return False


def user_explicitly_requests_evolve(user_request: str, conversation_history: list) -> bool:
    """判断用户是否明确要求使用 evolve 流程
    
    Args:
        user_request: 用户输入
        conversation_history: 对话历史
        
    Returns:
        bool: 是否明确要求 evolve
    """
    user_request_lower = user_request.lower()
    
    # 只有明确提到 "evolve" 的才走 evolve 流程
    # 所有涉及"性能"的请求都走 kernel_verifier（性能测试）
    explicit_evolve_keywords = [
        "使用evolve", "用evolve", "evolve流程", "evolve优化", "evolve生成",
        "use evolve", "using evolve", "with evolve",
        "走evolve", "选evolve", "要evolve", "换evolve", "改用evolve",
        "切换到evolve", "改成evolve", "试试evolve"
    ]
    
    if any(kw in user_request_lower for kw in explicit_evolve_keywords):
        logger.info(f"User explicitly mentioned 'evolve' keyword")
        return True
    
    # 只保留纯优化关键词（不涉及性能），且需要组合才触发
    # 注意：删除了所有"性能"相关的关键词，避免与 kernel_verifier 冲突
    optimization_keywords = [
        "多轮优化", "迭代优化", "进化优化", "自动调优", "自动优化",
        "iterative optimization", "evolutionary optimization", "auto-tune",
        "多次迭代", "多轮迭代"
    ]
    
    # 只有明确要求"多轮优化"、"迭代优化"等才走 evolve
    # 单独的"优化"不够，必须是组合词
    if any(kw in user_request_lower for kw in optimization_keywords):
        logger.info(f"User requested iterative optimization, using evolve")
        return True
    
    # 检查对话历史中是否有明确要求
    for msg in conversation_history[-3:]:  # 检查最近3轮对话
        content = msg.get("content", "").lower()
        if any(kw in content for kw in explicit_evolve_keywords):
            logger.info(f"Found evolve keyword in conversation history")
            return True
    
    return False


def is_modification_request(user_request: str, has_previous_code: bool) -> bool:
    """判断用户输入是否是修改请求

    Args:
        user_request: 用户输入
        has_previous_code: 是否有之前的代码

    Returns:
        bool: 是否是修改请求
    """
    # 如果没有之前的 task_code，一定不是修改请求
    if not has_previous_code:
        return False
    
    # 修改相关的关键词
    modification_keywords = [
        "修改", "改成", "改为", "换成", "调整", "优化", "更新",
        "变成", "改一下", "换一下", "调成", "设为", "设成",
        "change", "modify", "update", "adjust", "alter", 
        "revise", "refine", "optimize", "improve"
    ]

    # 检查是否包含修改关键词
    user_request_lower = user_request.lower()
    contains_modification = any(kw in user_request_lower for kw in modification_keywords)

    if contains_modification:
        logger.info(f"Detected modification request, skipping intent classification")
        return True

    # 如果用户输入很短且包含"shape"、"size"等，可能是修改请求
    if len(user_request) < 50 and any(kw in user_request_lower for kw in ["shape", "size", "dim", "维度", "形状"]):
        logger.info(f"Detected potential modification request (short input with shape/size), skipping intent classification")
        return True

    return False


def simple_action_heuristic(state: Dict[str, Any], user_input: str) -> str:
    """简单的启发式规则来决定 action（作为 fallback）
    
    Args:
        state: 当前状态
        user_input: 用户输入
        
    Returns:
        str: 建议的action ('confirm', 'revise', 'retry', 'retry_sub_agent', 'cancel')
    """
    user_lower = user_input.lower()
    has_task_code = bool(state.get("task_code"))
    has_generated_code = bool(state.get("generated_code"))

    # 检查取消意图
    if any(kw in user_lower for kw in ['取消', '退出', '结束', 'cancel', 'quit', 'exit']):
        return 'cancel'

    # 已生成代码的情况
    if has_generated_code:
        # 明确提到 task/torch → retry
        if any(kw in user_lower for kw in ['task', 'torch', '任务', 'pytorch']):
            return 'retry'
        # 其他情况 → retry_sub_agent
        return 'retry_sub_agent'

    # 有 task_code 但未生成代码
    if has_task_code:
        # 检查确认意图
        if any(kw in user_lower for kw in ['确认', 'ok', 'yes', '好', '生成', 'generate']):
            return 'confirm'
        # 其他情况 → revise
        return 'revise'

    # 默认 revise
    return 'revise'


def format_agents_info_for_llm(agents_info: Dict[str, Dict[str, Any]]) -> str:
    """格式化子 Agent 信息供 LLM 理解
    
    Args:
        agents_info: 子Agent详细信息字典
        
    Returns:
        str: 格式化后的文本
    """
    lines = []
    for agent_name, info in agents_info.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"Agent: {info['name']}")
        lines.append(f"Description: {info['description']}")
        lines.append(f"\nWorkflow Steps:")
        for step in info['workflow_steps']:
            lines.append(f"  - {step}")
        lines.append(f"\nUse Cases:")
        for case in info['use_cases']:
            lines.append(f"  - {case}")
        lines.append(f"\nAdvantages:")
        for adv in info['advantages']:
            lines.append(f"  + {adv}")
        lines.append(f"\nLimitations:")
        for lim in info['limitations']:
            lines.append(f"  - {lim}")
        lines.append(f"\nPerformance: {info['performance']}")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)

