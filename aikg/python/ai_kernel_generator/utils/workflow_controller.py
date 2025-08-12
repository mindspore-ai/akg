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

import logging
from typing import Set, List, Dict, Any

logger = logging.getLogger(__name__)


class WorkflowController:
    """
    工作流控制器
    负责工作流的控制逻辑，包括违法检查、链重复检查等
    """

    @staticmethod
    def get_illegal_agent(step_count: int, max_step: int, current_agent_name: str,
                          agent_history: List[str], repeat_limits: Dict[str, Any],
                          agent_info: Dict[str, Any]) -> Set[str]:
        """
        根据yaml文件要求，检查操作上限和总步数上限
        返回违禁操作的agent集合

        Args:
            step_count: 当前步数
            max_step: 最大步数
            current_agent_name: 当前agent名称
            agent_history: agent执行历史
            repeat_limits: 重复次数限制配置
            agent_info: agent信息配置

        Returns:
            违禁agent集合
        """
        illegal_agents = set()

        # 检查总步数上限
        if step_count >= max_step:
            logger.info(f"Step count {step_count} exceeds max_step {max_step}")
            return set(agent_info.keys())  # 返回所有agent

        # 检查单个agent的连续重复次数限制
        if current_agent_name and agent_history:
            # 获取当前agent的限制（默认3次）
            if repeat_limits and 'single_agent' in repeat_limits:
                max_repeats = repeat_limits['single_agent'].get(current_agent_name, 3)
            else:
                max_repeats = 3

            consecutive_count = WorkflowController.count_consecutive_repeats(agent_history, current_agent_name)
            if consecutive_count >= max_repeats:
                logger.debug(
                    f"Agent {current_agent_name} consecutive count {consecutive_count} exceeds limit {max_repeats}")
                illegal_agents.add(current_agent_name)

        # 检查特定序列的重复次数限制（仅在有配置时检查）
        if repeat_limits:
            sequences = repeat_limits.get('sequences', {})
            for seq_name, seq_config in sequences.items():
                pattern = seq_config.get('pattern', [])
                max_repeats = seq_config.get('max_repeats', 0)

                if not pattern or not current_agent_name:
                    continue

                # 只有当前agent是序列的最后一个agent时才检查这个序列
                if current_agent_name != pattern[-1]:
                    continue

                # 统计当前序列的重复次数
                repeat_count = WorkflowController.count_sequence_repeats(agent_history, pattern)

                if repeat_count >= max_repeats:
                    logger.debug(f"Sequence {seq_name} repeat count {repeat_count} exceeds limit {max_repeats}")
                    # 如果超过限制，添加序列中的第一个agent到违禁列表
                    if pattern:
                        illegal_agents.add(pattern[0])

        logger.debug(f"Illegal agents: {illegal_agents}")
        return illegal_agents

    @staticmethod
    def count_consecutive_repeats(agent_history: List[str], agent_name: str) -> int:
        """
        统计指定agent在历史末尾的连续重复次数

        Args:
            agent_history: agent执行历史
            agent_name: 要检查的agent名称

        Returns:
            连续重复次数
        """
        if not agent_history or not agent_name:
            return 0

        count = 0
        # 从后往前计算连续重复次数
        for i in range(len(agent_history) - 1, -1, -1):
            if agent_history[i] == agent_name:
                count += 1
            else:
                break

        return count

    @staticmethod
    def count_sequence_repeats(agent_history: List[str], pattern: List[str]) -> int:
        """
        统计指定序列在agent历史末尾的连续重复次数
        检查历史末尾是否匹配重复的序列模式

        Args:
            agent_history: agent执行历史
            pattern: 要检查的序列模式

        Returns:
            重复次数
        """
        if not pattern or not agent_history:
            return 0

        pattern_length = len(pattern)
        history_length = len(agent_history)
        max_possible_repeats = history_length // pattern_length

        # 从最大可能重复数开始向下检查
        for repeat_count in range(max_possible_repeats, 0, -1):
            required_length = repeat_count * pattern_length
            if required_length <= history_length:
                # 检查历史末尾指定长度是否匹配重复的序列模式
                tail_segment = agent_history[-required_length:]
                expected_pattern = pattern * repeat_count

                if tail_segment == expected_pattern:
                    return repeat_count

        return 0

    @staticmethod
    def get_valid_next_agent(agent_name: str, agent_next_mapping: Dict[str, Set[str]],
                             step_count: int, max_step: int, current_agent_name: str,
                             agent_history: List[str], repeat_limits: Dict[str, Any],
                             agent_info: Dict[str, Any]) -> Set[str]:
        """
        根据yaml文件要求，获取next_agent_name的可选选项
        在可选选项中排除illegal_agent中的选项

        Args:
            agent_name: 当前agent名称
            agent_next_mapping: agent下一步映射
            step_count: 当前步数
            max_step: 最大步数
            current_agent_name: 当前agent名称
            agent_history: agent执行历史
            repeat_limits: 重复次数限制配置
            agent_info: agent信息配置

        Returns:
            可能的下一个agent集合
        """
        # 获取当前agent的可能下一步
        possible_next = agent_next_mapping.get(agent_name, set())

        # 获取违禁agent
        illegal_agents = WorkflowController.get_illegal_agent(
            step_count, max_step, current_agent_name, agent_history,
            repeat_limits, agent_info
        )

        # 使用set的差集运算符排除illegal_agent中的选项
        valid_next_agents = possible_next - illegal_agents

        logger.debug(
            f"Agent {agent_name} -> possible: {possible_next}, illegal: {illegal_agents}, valid: {valid_next_agents}")
        return valid_next_agents
