# Copyright 2026 Huawei Technologies Co., Ltd
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

"""通用路由工具函数

提供领域无关的路由决策工具，可被任何 LangGraph 工作流使用。
"""

from typing import Set, Dict
import logging

logger = logging.getLogger(__name__)


def check_step_limit(step_count: int, max_step: int) -> bool:
    """检查是否超过步数限制
    
    Args:
        step_count: 当前步数
        max_step: 最大步数
        
    Returns:
        True 表示已超限
    """
    return step_count >= max_step


def check_agent_repeat_limit(agent_history: list, agent_name: str, max_repeats: int = 3) -> bool:
    """检查 agent 是否超过连续重复次数限制
    
    Args:
        agent_history: Agent 执行历史列表
        agent_name: 要检查的 agent 名称
        max_repeats: 最大连续重复次数
        
    Returns:
        True 表示已超限，应禁止该 agent
    """
    if not agent_history:
        return False
    
    consecutive_count = 0
    for agent in reversed(agent_history):
        if agent == agent_name:
            consecutive_count += 1
        else:
            break
    
    return consecutive_count >= max_repeats


def get_illegal_agents(step_count: int, max_step: int, 
                       agent_history: list, repeat_limits: Dict[str, int]) -> Set[str]:
    """获取被禁止的 agent 集合
    
    综合检查步数限制和各 agent 的重复限制。
    
    Args:
        step_count: 当前步数
        max_step: 最大步数
        agent_history: Agent 执行历史
        repeat_limits: 各 agent 的重复次数限制，如 {"coder": 3, "designer": 2}
        
    Returns:
        被禁止的 agent 名称集合。如果返回 {"*"}，表示全部禁止。
    """
    illegal = set()
    
    # 检查总步数上限
    if check_step_limit(step_count, max_step):
        logger.info(f"Step count {step_count} exceeds max_step {max_step}, all agents forbidden")
        return {"*"}  # 特殊标记：全部禁止
    
    # 检查各 agent 的重复限制
    for agent_name, limit in repeat_limits.items():
        if check_agent_repeat_limit(agent_history, agent_name, limit):
            logger.debug(f"Agent '{agent_name}' exceeds repeat limit {limit}")
            illegal.add(agent_name)
    
    return illegal


def filter_valid_agents(possible_agents: Set[str], illegal_agents: Set[str]) -> Set[str]:
    """从可能的 agent 集合中过滤掉被禁止的
    
    Args:
        possible_agents: 可能的 agent 集合
        illegal_agents: 被禁止的 agent 集合
        
    Returns:
        有效的 agent 集合
    """
    if "*" in illegal_agents:
        return set()  # 全部禁止
    
    return possible_agents - illegal_agents

