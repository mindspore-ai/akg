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
from typing import Dict, Any, Optional
from ai_kernel_generator import get_project_root
from .common_utils import ParserFactory, load_yaml

logger = logging.getLogger(__name__)


def _convert_to_internal_format(parser_definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    将解析器定义格式转换为ParserFactory期望的格式

    Args:
        parser_definition: 解析器定义

    Returns:
        转换后的解析器配置
    """
    parser_config = {

        'output_fields': {}
    }

    # 处理字段定义
    output_fields = parser_definition.get('output_fields', {})

    for field_name, field_config in output_fields.items():
        if isinstance(field_config, dict):
            # 标准格式：包含详细配置
            parser_config['output_fields'][field_name] = {
                'field_type': field_config.get('field_type', 'str'),
                'mandatory': field_config.get('mandatory', True),
                'field_description': field_config.get('field_description', f'{field_name}字段')
            }
        else:
            # 简单格式：只有类型
            parser_config['output_fields'][field_name] = {
                'field_type': str(field_config),
                'mandatory': True,
                'field_description': f'{field_name}字段'
            }

    return parser_config


def create_step_parser(step_name: str, workflow_config_path: Optional[str] = None):
    """
    为特定工作流步骤创建解析器

    Args:
        step_name: 步骤名称（如'designer', 'coder'）
        workflow_config_path: workflow配置文件路径，可选

    Returns:
        解析器实例，如果该步骤不需要解析器则返回None
    """
    try:
        # 加载workflow配置
        workflow_config = load_yaml(workflow_config_path)

        # 获取agent_info
        agent_info = workflow_config.get('agent_info', {})
        if not agent_info:
            raise ValueError("No 'agent_info' found in workflow config")

        # 检查步骤是否存在
        if step_name not in agent_info:
            raise ValueError(f"Step '{step_name}' not found in agent_info")

        step_config = agent_info[step_name]

        # 检查是否有output_format配置
        output_format = step_config.get('output_format')
        if not output_format:
            logger.info(f"Step '{step_name}' has no output_format, returning None (no parser needed)")
            return None

        parser_name = output_format.get('parser_name')
        if not parser_name:
            logger.warning(f"No parser_name found in step '{step_name}'")
            return None

        # 检查是否有parser_definition
        parser_definition = output_format.get('parser_definition')
        if not parser_definition:
            logger.warning(f"No parser_definition found in step '{step_name}'")
            return None

        # 转换为内部格式并注册解析器
        parser_config = _convert_to_internal_format(parser_definition)
        ParserFactory.register_parser(parser_name, parser_config)
        return ParserFactory.get_parser(parser_name)

    except Exception as e:
        logger.error(f"Failed to create parser for step '{step_name}': {e}")
        raise
