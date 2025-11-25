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
from typing import List, Dict, Any, Optional, Set, Tuple
from ai_kernel_generator.utils.common_utils import ParserFactory
from ai_kernel_generator.utils.parser_registry import create_step_parser
from ai_kernel_generator.core.trace import Trace

logger = logging.getLogger(__name__)


class ResultProcessor:
    """
    结果处理器
    负责解析agent结果、更新任务信息、查找代码等功能
    """

    @staticmethod
    def parse_and_update_code(agent_name: str, result: str, task_info: Dict[str, Any],
                              agent_parser, trace: Trace, agent_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        解析agent结果并更新任务信息和保存解析后的代码

        Args:
            agent_name: agent名称
            result: 原始json结果
            task_info: 任务信息字典（会被修改）
            agent_parser: agent解析器
            trace: trace实例

        Returns:
            是否解析成功
        """
        try:
            if not agent_parser or not result:
                logger.warning(f"No parser or empty result for {agent_name}")
                return False

            # 解析结果
            parsed_result = ParserFactory.robust_parse(result, agent_parser)
            if not parsed_result:
                logger.warning(f"Failed to parse result for {agent_name}")
                return False

            params_to_save = []

            # 从workflow配置中获取该agent定义的输出字段
            agent_config = agent_info.get(agent_name, {})
            output_format = agent_config.get('output_format', {})
            parser_definition = output_format.get('parser_definition', {})
            output_fields = parser_definition.get('output_fields', {})

            # 根据workflow.yaml中定义的字段提取数据
            field_names = list(output_fields.keys())

            # 如果只有一个字段，直接使用
            if len(field_names) == 1:
                field_name = field_names[0]
                field_value = getattr(parsed_result, field_name, '') or ''
                if field_value:
                    # 统一存储为{agent_name}_code格式
                    task_info_key = f"{agent_name}_code"
                    task_info[task_info_key] = field_value
                    # 添加到保存列表
                    params_to_save.append(("code", field_value))

            # 如果有多个字段，处理每个字段
            elif len(field_names) > 1:
                code_fields = [name for name in field_names if "code" in name.lower()]
                if not code_fields:
                    # 如果没有找到包含"code"的字段，报错
                    logger.error(f"Agent '{agent_name}' has multiple output fields {field_names} but none contains 'code'. "
                                 f"Please either use a single output field or name one of them with 'code'.")
                    return False

                # 处理所有字段
                for field_name in field_names:
                    field_value = getattr(parsed_result, field_name, '') or ''
                    if field_value:
                        if "code" in field_name.lower():
                            # 包含"code"的字段保存为{agent_name}_code
                            task_info[f"{agent_name}_code"] = field_value
                            # 添加到保存列表（主要字段）
                            params_to_save.append(("code", field_value))
                        # 所有字段保存为{agent_name}_{field_name}
                        task_info[f"{agent_name}_{field_name}"] = field_value
                        # 也添加到保存列表
                        params_to_save.append((field_name, field_value))

            # 保存解析后的代码到文件
            if params_to_save:
                trace.save_parsed_code(agent_name, params_to_save)

            logger.debug(f"Parsed and updated {agent_name} fields: {len(params_to_save)} fields")
            return True

        except Exception as e:
            logger.error(f"Failed to parse and update code for {agent_name}: {e}")
            return False

    @staticmethod
    def update_verifier_result(result: str, error_log: str, task_info: Dict[str, Any], profile_res: Optional[dict] = None) -> None:
        """
        更新verifier结果

        Args:
            result: verifier结果
            error_log: 错误日志
            task_info: 任务信息字典（会被修改）
            profile_res: 性能分析结果字典，包含：
                - gen_time: 生成代码执行时间（微秒）
                - base_time: 基准代码执行时间（微秒）
                - speedup: 加速比
                - autotune_summary: autotune配置详情（可选，仅triton+ascend）
        """
        try:
            # 解析verifier结果
            if result == "True" or result is True:
                task_info['verifier_result'] = True
            elif result == "False" or result is False:
                task_info['verifier_result'] = False

            # 直接更新verifier_error字段
            task_info['verifier_error'] = error_log or ''

            if profile_res:
                task_info['profile_res'] = profile_res

            logger.debug(f"Updated verifier result: {task_info['verifier_result']}")

        except Exception as e:
            logger.error(f"Failed to update verifier result: {e}")

    @staticmethod
    def get_agent_parser(agent_name: str, workflow_config_path: str,
                         agent_parsers: Dict[str, Any]) -> Any:
        """
        为指定的agent获取解析器（带缓存）

        Args:
            agent_name: agent名称
            workflow_config_path: workflow配置路径
            agent_parsers: 解析器缓存字典

        Returns:
            解析器实例或None
        """
        if agent_name not in agent_parsers:
            try:
                parser = create_step_parser(agent_name, workflow_config_path)
                agent_parsers[agent_name] = parser
                if parser is None:
                    logger.info(f"Agent '{agent_name}' does not need a parser")
                else:
                    logger.debug(f"Cached parser for agent '{agent_name}'")
            except Exception as e:
                logger.error(f"Failed to create parser for agent '{agent_name}': {e}")
                agent_parsers[agent_name] = None

        return agent_parsers[agent_name]

    @staticmethod
    def parse_conductor_decision(content: str, conductor_parser, valid_next_agents: Set[str]) -> Tuple[Optional[str], str]:
        """解析Conductor的LLM决策结果

        Args:
            content: LLM返回的原始内容
            conductor_parser: Conductor解析器
            valid_next_agents: 有效的下一个agent集合

        Returns:
            Tuple[Optional[str], str]: (决策的agent名称, suggestion信息)
        """
        try:
            parsed_result = conductor_parser.parse(content)
            if parsed_result and hasattr(parsed_result, 'decision'):
                decision = parsed_result.decision.strip().lower()
                suggestion = getattr(parsed_result, 'error_and_suggestion', '')

                # 验证决策是否有效
                for agent in valid_next_agents:
                    if agent.lower() == decision:
                        logger.info(f"LLM decided next agent: {agent}" +
                                    (f" (suggestion: {suggestion})" if suggestion else ""))
                        return agent, suggestion

                logger.warning(f"LLM decision '{decision}' not in valid agents {valid_next_agents}")
                return None, suggestion
            else:
                logger.warning(f"Failed to parse LLM decision: {content}")
                return None, ""

        except Exception as parse_error:
            logger.warning(f"LLM decision parsing error: {parse_error}, content: {content}")
            return None, ""
