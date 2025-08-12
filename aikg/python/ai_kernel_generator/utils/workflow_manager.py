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

import os
import logging
from typing import Dict, Any, Optional, Set
from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import load_yaml

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    工作流配置管理器
    负责加载和管理workflow.yaml配置
    """

    @staticmethod
    def resolve_workflow_config_path(workflow_name_or_path: Optional[str] = None) -> str:
        """
        解析workflow配置路径，支持多种输入格式

        Args:
            workflow_name_or_path: 可以是：
                - None: 使用默认配置
                - 文件名: "coder_only_workflow" -> "config/coder_only_workflow.yaml"
                - 完整路径: "config/xxx.yaml" 或 "/path/to/xxx.yaml"

        Returns:
            str: 解析后的完整路径
        """
        # 如果已经是完整的文件路径（包含.yaml或.yml）
        if workflow_name_or_path.endswith(('.yaml', '.yml')):
            # 如果是绝对路径，直接返回
            if os.path.isabs(workflow_name_or_path):
                return workflow_name_or_path
            # 如果是相对路径，相对于项目根目录
            return os.path.join(get_project_root(), workflow_name_or_path)

        # 如果只是文件名（不带扩展名），自动添加路径和扩展名
        workflow_filename = f"{workflow_name_or_path}.yaml"
        config_path = os.path.join("config", workflow_filename)
        return os.path.join(get_project_root(), config_path)

    @staticmethod
    def load_workflow_config(workflow_config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        从workflow.yaml文件读取配置信息

        Args:
            workflow_config_path: workflow配置文件路径，可选

        Returns:
            包含agent_info, limitation_info等的配置字典
        """
        try:
            config = load_yaml(workflow_config_path)

            # 获取agent_info和limitation_info
            agent_info = config.get('agent_info', {})
            if not agent_info:
                raise ValueError("No 'agent_info' found in workflow config")

            limitation_info = config.get('limitation_info', {})

            # 处理必须设置的项
            required_settings = limitation_info.get('required', {})
            max_step = required_settings.get('max_step')
            if max_step is None:
                raise ValueError("Missing required setting 'max_step' in limitation_info.required")

            # 处理可选设置的项
            optional_settings = limitation_info.get('optional', {})
            repeat_limits = optional_settings.get('repeat_limits', {})

            # 获取start_agent
            start_agent = config.get('start_agent')
            if start_agent is None:
                raise ValueError("Missing required 'start_agent' in workflow config")

            # 获取mandatory_llm_analysis（可选）
            mandatory_llm_analysis = config.get('mandatory_llm_analysis', [])

            # 解析各agent的可能下一步
            agent_next_mapping = {}
            for agent_name, agent_config in agent_info.items():
                if isinstance(agent_config, dict) and 'possible_next_agent' in agent_config:
                    agent_next_mapping[agent_name] = set(agent_config['possible_next_agent'])

            logger.debug(
                f"Loaded workflow config: agents={len(agent_info)}, max_steps={max_step}, start={start_agent}")

            return {
                'agent_info': agent_info,
                'limitation_info': limitation_info,
                'start_agent': start_agent,
                'max_step': max_step,
                'repeat_limits': repeat_limits,
                'agent_next_mapping': agent_next_mapping,
                'mandatory_llm_analysis': mandatory_llm_analysis
            }

        except Exception as e:
            logger.error(f"Failed to load workflow config: {e}")
            raise

    @staticmethod
    def initialize_task_info_fields(agent_info: Dict[str, Any], op_name: str,
                                    task_id: str, dsl: str, task_desc: str = "",
                                    base_doc: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        根据agent_info动态初始化task_info字段

        Args:
            agent_info: workflow.yaml中的agent_info配置
            op_name: 算子名称
            task_id: 任务ID 
            dsl: 实现类型
            task_desc: 任务描述
            base_doc: 基础文档字典，包含各种API文档等

        Returns:
            初始化的task_info字典
        """
        task_info = {
            'op_name': op_name,
            'task_id': task_id,
            'dsl': dsl,
            'task_desc': task_desc
        }

        if base_doc:
            task_info.update(base_doc)

        # 根据agent_info中的output_fields动态添加字段
        for agent_name, agent_config in agent_info.items():
            output_fields = agent_config.get('output_format', {}).get('parser_definition', {}).get('output_fields', {})
            for field_name in output_fields.keys():
                task_info[f"{agent_name}_{field_name}"] = ""

        # 添加verifier特殊字段（因为verifier没有parser_definition）
        task_info['verifier_result'] = False
        task_info['verifier_error'] = ""

        return task_info
