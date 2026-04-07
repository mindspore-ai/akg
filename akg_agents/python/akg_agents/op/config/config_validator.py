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
from akg_agents.utils.common_utils import load_yaml
from akg_agents import get_project_root
from akg_agents.op.utils.config_utils import normalize_dsl

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    配置校验器类，用于校验配置文件的合法性。
    目前支持的配置项：
        - log_dir: 日志目录，默认为~/tmp。
        - docs_dir: 文档目录配置。
    """

    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)

    def validate_log_dir(self):
        if 'log_dir' not in self.config:
            raise ValueError("配置文件中缺少 log_dir 字段")

        root_dir = os.path.expanduser(self.config['log_dir'])
        self.config['log_dir'] = Path(root_dir) / f"Task_{next(tempfile._get_candidate_names())}"

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
            self.validate_log_dir()
            self.validate_docs_dir()
        except ValueError as e:
            raise ValueError(f"配置校验失败：{str(e)}")


def load_config(dsl="", config_path: Optional[str] = None, backend: Optional[str] = None, workflow: Optional[str] = "coder_only"):
    """
    加载并验证配置文件

    Args:
        dsl: 领域特定语言类型，用于选择默认配置文件。如果为"triton"，需要提供backend参数进行自动转换
        config_path: 配置文件路径，如果不提供则根据dsl和(可选)workflow选择默认配置
        backend: 硬件后端名称(ascend/cuda)，用于自动转换dsl="triton"为triton_cuda或triton_ascend
        workflow: 工作流类型(coder_only/default/kernelgen_only/evolve)，用于选择特定配置，默认为coder_only

    Returns:
        dict: 验证后的配置

    Raises:
        ValueError: 如果没有config_path且根据dsl找不到默认配置文件，或dsl="triton"但未提供backend
    """
    # 1. 规范化DSL（自动转换triton）
    if dsl:
        normalized_dsl = normalize_dsl(dsl, backend or "")
        dsl = normalized_dsl  # 使用规范化后的DSL
    
    # 2. 有config_path时直接使用config_path
    if config_path:
        final_config_path = Path(config_path)
    else:
        # 3. 没有config_path时，根据dsl和workflow选择配置
        # 优先级：{dsl}_{workflow}_config.yaml > default_{dsl}_config.yaml
        if workflow and dsl:
            workflow_suffix = workflow.replace("_workflow", "")
            final_config_path = Path(__file__).parent / f"{dsl}_{workflow_suffix}_config.yaml"
            if not final_config_path.exists():
                final_config_path = Path(__file__).parent / f"default_{dsl}_config.yaml"
        elif dsl:
            final_config_path = Path(__file__).parent / f"default_{dsl}_config.yaml"
        else:
            raise ValueError("必须提供 dsl 或 config_path")

    # 4. 检查配置文件是否存在，不存在就抛出错误
    if not final_config_path.exists():
        raise ValueError(f"No config found for dsl '{dsl}' and workflow '{workflow}'. "
                         f"Please provide config_path like load_config('/path-to-config/xxx_config.yaml') "
                         f"or ensure config exists at: {final_config_path}")

    validator = ConfigValidator(str(final_config_path))
    validator.validate_all()
    return validator.config
