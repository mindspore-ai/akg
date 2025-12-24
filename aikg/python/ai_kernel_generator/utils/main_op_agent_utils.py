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


def extract_torch_task_code(user_input: str) -> Optional[str]:
    """
    从用户输入中提取 KernelBench 格式的 torch task 代码
    
    KernelBench 格式必须包含：
    - import torch
    - class Model(nn.Module)
    - def forward(self, ...)
    - def get_inputs()
    - def get_init_inputs()  ← 必须有，即使返回空列表
    
    Args:
        user_input: 用户输入
        
    Returns:
        提取的 torch task 代码，如果没有则返回 None
    """
    # 检查是否包含必要的 KernelBench 标志（5个必需字段）
    required_patterns = [
        r"import\s+torch",
        r"class\s+Model\s*\(",
        r"def\s+forward\s*\(",
        r"def\s+get_inputs\s*\(",
        r"def\s+get_init_inputs\s*\(",  # 🆕 必须有
    ]
    
    # 检查是否所有必要模式都存在
    if not all(re.search(pattern, user_input, re.MULTILINE) for pattern in required_patterns):
        return None
    
    # 尝试提取代码块
    # 方法1: 查找 ```python 代码块
    code_block_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", user_input, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
        # 验证提取的代码是否符合要求
        if all(re.search(pattern, code, re.MULTILINE) for pattern in required_patterns):
            logger.info("Extracted torch task code from markdown code block")
            return code
    
    # 方法2: 直接提取整段代码（没有代码块标记）
    # 查找从 import torch 开始，到最后一个必需函数结束
    import_match = re.search(r"import\s+torch", user_input)
    if not import_match:
        return None
    
    start_pos = import_match.start()
    
    # 找到最后一个必需函数的结束位置
    # 查找 def get_init_inputs() 之后的第一个 return 语句或函数结束
    get_init_inputs_match = re.search(r"def\s+get_init_inputs\s*\([^)]*\)\s*:", user_input[start_pos:])
    if not get_init_inputs_match:
        return None
    
    # 从 get_init_inputs 开始查找，找到这个函数的结束位置
    func_start = start_pos + get_init_inputs_match.end()
    
    # 查找函数体（缩进的代码或 return 语句）
    # 匹配：return [...] 或 return [] 或多行的函数体
    func_body_match = re.search(r"return\s+\[.*?\]", user_input[func_start:], re.DOTALL)
    if func_body_match:
        # 找到 return 语句的结束位置
        code_end = func_start + func_body_match.end()
    else:
        # 如果没找到 return，尝试找到函数体结束（下一个非缩进行或文件结束）
        remaining = user_input[func_start:]
        # 跳过空白，找到下一个非函数体的内容
        body_end_match = re.search(r"\n(?=\S)", remaining)
        if body_end_match:
            code_end = func_start + body_end_match.start()
        else:
            code_end = len(user_input)
    
    # 提取代码
    code = user_input[start_pos:code_end].strip()
    
    # 进一步清理：移除代码后面可能的非代码文本
    # 查找代码后是否有中文或明显的自然语言（作为结束标记）
    # 在最后一个 ] 或 ) 之后，如果有中文字符，就截断
    last_bracket = max(code.rfind(']'), code.rfind(')'))
    if last_bracket > 0:
        # 从最后一个括号之后查找中文或请求性文字
        after_bracket = code[last_bracket + 1:]
        chinese_match = re.search(r'[\u4e00-\u9fff]', after_bracket)
        if chinese_match:
            # 找到中文，截断到最后一个括号
            code = code[:last_bracket + 1].strip()
    
    # 验证提取的代码是否符合要求
    if all(re.search(pattern, code, re.MULTILINE) for pattern in required_patterns):
        # 如果代码是一行连续的，在关键位置添加换行
        if '\n' not in code or code.count('\n') < 3:
            # 步骤1：在每个 import 后添加换行
            code = re.sub(r'\b(import\s+[\w.]+(?:\s+as\s+\w+)?)\s+', r'\1\n', code)
            
            # 步骤2：在 class 定义前添加两个换行
            code = re.sub(r'\s*(class\s+)', r'\n\n\1', code)
            
            # 步骤3：在 class 内的方法之间添加换行
            # 先处理 __init__ 方法
            code = re.sub(r'(class\s+\w+[^:]*:)\s*(def\s+__init__)', r'\1\n    \2', code)
            # __init__ 方法的冒号后换行
            code = re.sub(r'(def\s+__init__[^:]*:)\s*', r'\1\n        ', code)
            # 处理 __init__ 之后的 forward 等方法
            code = re.sub(r'(\)\s+)(def\s+\w+\s*\()', r')\n    \2', code)
            
            # 步骤4：在顶层函数 def 前添加换行（class 外的函数）
            # 匹配 "] def " 或 ") def " 模式（在 class 之后）
            code = re.sub(r'([\]\)])\s*(def\s+(?:get_inputs|get_init_inputs))', r'\1\n\n\2', code)
            
            # 步骤5：在方法体内的语句之间添加换行
            # super().__init__() 后换行
            code = re.sub(r'(super\(\).__init__\(\))\s+', r'\1\n        ', code)
            # self.xxx = 后，如果还有 def，添加换行
            code = re.sub(r'(self\.\w+\s*=\s*[^\n]+?)\s+(def\s+)', r'\1\n    \2', code)
            # forward 等方法的冒号后换行
            code = re.sub(r'(def\s+forward[^:]*:)\s*', r'\1\n        ', code)
            
            # 步骤6：顶层函数的 return 语句处理
            code = re.sub(r'(get_inputs\(\):)\s*(return\s+)', r'\1\n    \2', code)
            code = re.sub(r'(get_init_inputs\(\):)\s*(return\s+)', r'\1\n    \2', code)
            
            # 清理多余的空行
            code = re.sub(r'\n{3,}', r'\n\n', code)
            code = code.strip()
        
        logger.info(f"Extracted torch task code from plain text, length: {len(code)} chars")
        return code
    
    return None


def has_torch_task_code(user_input: str) -> bool:
    """
    判断用户输入中是否包含 KernelBench 格式的 torch task 代码
    
    Args:
        user_input: 用户输入
        
    Returns:
        是否包含 torch task 代码
    """
    return extract_torch_task_code(user_input) is not None


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
        "operator_dev",  # 🆕 添加简短形式
        "code_generation",
        "modification",
        "optimization",
        "unclear"
    ]
    
    # 🆕 更宽松的判断：只要包含 "operator" 关键词，就认为是相关的
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
        "高性能算子", "超高性能",  # 🆕 增加高性能关键词
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
    简单的启发式规则判断用户动作
    
    Args:
        state: 当前状态
        user_input: 用户输入
        
    Returns:
        tuple: (action, is_new_operator, has_provided_task_code, is_complete_code, extracted_task_code, extracted_op_name, extracted_op_description)
    """
    user_input_lower = user_input.lower()
    
    # 确认关键词
    confirm_keywords = ["确认", "ok", "yes", "好的", "可以", "继续", "confirm"]
    if any(kw in user_input_lower for kw in confirm_keywords):
        return "confirm", False, False, False, '', '', ''
    
    # 取消关键词
    cancel_keywords = ["取消", "退出", "结束", "再见", "cancel", "quit", "exit"]
    if any(kw in user_input_lower for kw in cancel_keywords):
        return "cancel", False, False, False, '', '', ''
    
    # 重试关键词
    retry_keywords = ["重新", "再试", "retry", "重做"]
    if any(kw in user_input_lower for kw in retry_keywords):
        if state.get("generated_code"):
            return "retry_sub_agent", False, False, False, '', '', ''
        else:
            return "retry", False, False, False, '', '', ''
    
    # 默认：修改
    return "revise", False, False, False, '', '', ''


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
