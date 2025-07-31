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
import os
import tempfile
from pathlib import Path
from typing import Optional
from ai_kernel_generator.utils.common_utils import load_yaml
from ai_kernel_generator import get_project_root

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    配置校验器类，用于校验配置文件的合法性。
    目前支持的配置项：
        - agent_model_config: 大模型名称，支持的模型名称请参考llm_config.yaml文件。
        - log_dir: 日志目录，默认为~/tmp。
    """

    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)

    def validate_llm_models(self):
        if 'agent_model_config' not in self.config:
            raise ValueError("配置文件中缺少 agent_model_config 字段")

        user_model_config = self.config['agent_model_config']

        # 从llm_config.yaml加载所有预设模型名称
        llm_config_path = Path(get_project_root()) / "core" / "llm" / "llm_config.yaml"
        llm_config = load_yaml(llm_config_path)
        valid_presets = llm_config.keys()

        # 检查模型预设名称合法性
        invalid_models = [f"{agent}:{model}" for agent, model in user_model_config.items()
                          if model not in valid_presets]

        if invalid_models:
            raise ValueError(f"非法的模型名称配置: {', '.join(invalid_models)}。请检查llm_config.yaml中的预设名称")

        self.config['agent_model_config'] = user_model_config

    def validate_log_dir(self):
        if 'log_dir' not in self.config:
            raise ValueError("配置文件中缺少 log_dir 字段")

        root_dir = os.path.expanduser(self.config['log_dir'])
        self.config['log_dir'] = Path(root_dir) / f"Task_{next(tempfile._get_candidate_names())}"

    def validate_workflow_config_path(self):
        if 'workflow_config_path' not in self.config:
            logger.warning("配置文件中缺少 workflow_config_path 字段")

        workflow_config_path = self.config['workflow_config_path']
        # 如果是相对路径，检查文件是否存在
        if not os.path.isabs(workflow_config_path):
            full_path = Path(get_project_root()) / workflow_config_path
        else:
            full_path = Path(workflow_config_path)

        if not full_path.exists():
            raise ValueError(f"workflow_config_path 指定的文件不存在: {full_path}")

    def validate_docs_dir(self):
        if 'docs_dir' not in self.config:
            raise ValueError("配置文件中缺少 docs_dir 字段")

        docs_dir_config = self.config['docs_dir']
        if not isinstance(docs_dir_config, dict):
            raise ValueError("docs_dir 必须是一个字典，包含各个agent类型的文档目录配置")

        # 检查每个文档目录是否存在
        for agent_type, docs_dir in docs_dir_config.items():
            if not os.path.isabs(docs_dir):
                full_path = Path(get_project_root()) / docs_dir
            else:
                full_path = Path(docs_dir)

            if not full_path.exists():
                raise ValueError(f"docs_dir 中 {agent_type} 指定的目录不存在: {full_path}")

    def validate_all(self):
        try:
            self.validate_llm_models()
            self.validate_log_dir()
            self.validate_workflow_config_path()
            self.validate_docs_dir()
        except ValueError as e:
            logger.error(f"配置校验失败：{str(e)}")
            raise


def load_config(dsl="", config_path: Optional[str] = None):
    """
    加载并验证配置文件

    Args:
        dsl: 领域特定语言类型，用于选择默认配置文件
        config_path: 配置文件路径，如果不提供则根据dsl选择默认配置

    Returns:
        dict: 验证后的配置

    Raises:
        ValueError: 如果没有config_path且根据dsl找不到默认配置文件
    """
    # 1. 有config_path时直接使用config_path
    if config_path:
        final_config_path = config_path
    else:
        # 2. 没有config_path时，根据dsl选择默认配置
        final_config_path = Path(__file__).parent / f"default_{dsl}_config.yaml"

    # 3. 检查默认配置文件是否存在，不存在就抛出错误
    if not final_config_path.exists():
        raise ValueError(f"No default config found for dsl '{dsl}'. "
                            f"Please provide config_path like load_config('/path-to-config/xxx_config.yaml') "
                            f"or ensure default config exists at: {final_config_path}")

    validator = ConfigValidator(final_config_path)
    validator.validate_all()
    return validator.config
