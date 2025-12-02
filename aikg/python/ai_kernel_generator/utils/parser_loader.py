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

"""Parser loader - independent from workflow.yaml"""

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


def _get_parser_config_path(config_path: Optional[str] = None) -> str:
    """获取 parser 配置文件路径
    
    Args:
        config_path: 自定义配置文件路径（可选）
        
    Returns:
        配置文件完整路径
    """
    if config_path:
        if os.path.isabs(config_path):
            return config_path
        else:
            # 相对路径，相对于项目根目录
            return os.path.join(get_project_root(), config_path)
    
    # 默认路径
    module = __import__('ai_kernel_generator', fromlist=[''])
    module_path = os.path.dirname(os.path.abspath(str(module.__file__)))
    default_path = os.path.join(module_path, "config", "parser_config.yaml")
    return default_path


def create_agent_parser(agent_name: str, parser_config_path: Optional[str] = None):
    """
    为特定 Agent 创建解析器（独立于 workflow.yaml）
    
    Args:
        agent_name: Agent 名称（如 'designer', 'coder'）
        parser_config_path: parser 配置文件路径，可选（默认使用 config/parser_config.yaml）
    
    Returns:
        解析器实例，如果该 Agent 不需要解析器则返回 None
    """
    try:
        # 获取配置文件路径
        config_path = _get_parser_config_path(parser_config_path)
        
        # 加载 parser 配置
        parser_config = load_yaml(config_path)
        
        # 获取 parsers 配置
        parsers = parser_config.get('parsers', {})
        if not parsers:
            raise ValueError("No 'parsers' found in parser config")
        
        # 检查 Agent 是否存在
        if agent_name not in parsers:
            logger.info(f"Agent '{agent_name}' has no parser configuration, returning None (no parser needed)")
            return None
        
        agent_parser_config = parsers[agent_name]
        
        # 检查是否有 parser_definition
        parser_definition = agent_parser_config.get('parser_definition')
        if not parser_definition:
            logger.warning(f"No parser_definition found for agent '{agent_name}'")
            return None
        
        parser_name = agent_parser_config.get('parser_name', f"{agent_name}_parser")
        
        # 转换为内部格式并注册解析器
        parser_config_internal = _convert_to_internal_format(parser_definition)
        ParserFactory.register_parser(parser_name, parser_config_internal)
        return ParserFactory.get_parser(parser_name)
    
    except Exception as e:
        logger.error(f"Failed to create parser for agent '{agent_name}': {e}")
        raise


# 向后兼容：保持旧的函数名（但使用新实现）
def create_step_parser(step_name: str, workflow_config_path: Optional[str] = None):
    """
    向后兼容函数：为特定工作流步骤创建解析器
    
    注意：此函数现在从 parser_config.yaml 读取，不再依赖 workflow_config_path
    workflow_config_path 参数保留用于向后兼容，但会被忽略
    
    Args:
        step_name: 步骤名称（如 'designer', 'coder'）
        workflow_config_path: 已废弃，保留用于向后兼容
    
    Returns:
        解析器实例，如果该步骤不需要解析器则返回 None
    """
    if workflow_config_path:
        logger.warning(f"workflow_config_path parameter is deprecated for create_step_parser. "
                     f"Using parser_config.yaml instead. Ignoring: {workflow_config_path}")
    
    return create_agent_parser(step_name)

