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
DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"
class ConfigValidator:
    """
    配置校验器类，用于校验配置文件的合法性。
    目前支持的配置项：
        - agent_model_config: 大模型名称，支持的模型名称请参考llm_config.yaml文件。
        - log_dir: 日志目录，默认为~/tmp。
    """
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        self.default_config = load_yaml(DEFAULT_CONFIG)

    def validate_llm_models(self):
        # 从默认配置中获取支持的模型名称字典，这里假设默认配置中包含了agent_model的默认设置，你可以根据实际情况进行调整。
        supported_model_config = self.default_config['agent_model_config']
        
        if 'agent_model_config' not in self.config:
            self.config['agent_model_config'] = supported_model_config
            return
        user_model_config = self.config['agent_model_config']
        
        
        invalid_agents = [k for k in user_model_config if k not in supported_model_config.keys()]
        if invalid_agents:
            raise ValueError(f"非法的模型角色配置: {', '.join(invalid_agents)}。合法角色应为: {', '.join(sorted(supported_model_config.keys()))}")

        # 补充用户未配置的agent
        for agent in supported_model_config:
            if agent not in user_model_config:
                user_model_config[agent] = supported_model_config[agent]
        
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
        default_log_dir = self.default_config['log_dir']
        if 'log_dir' not in self.config:
            self.config['log_dir'] = default_log_dir
        root_dir = os.path.expanduser(self.config['log_dir'])
        self.config['log_dir'] = Path(root_dir) / f"Task_{next(tempfile._get_candidate_names())}"

    def validate_all(self):
        try:
            self.validate_llm_models()
            self.validate_log_dir()
        except ValueError as e:
            logger.error(f"配置校验失败：{str(e)}")
            raise
        
def load_config(config_path: Optional[str] = None):
    config_path = config_path or DEFAULT_CONFIG
    validator = ConfigValidator(config_path)
    validator.validate_all()
    return validator.config