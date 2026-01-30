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

"""算子工作流基类

继承通用 BaseWorkflow，添加算子生成场景的专用逻辑。
"""

from akg_agents.core_v2.langgraph_base.base_workflow import BaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.utils.common_utils import get_prompt_path
from jinja2 import Template
import os
import logging

logger = logging.getLogger(__name__)


class OpBaseWorkflow(BaseWorkflow[KernelGenState]):
    """算子工作流基类
    
    继承通用 BaseWorkflow，添加算子生成场景的专用属性：
    - agents: 算子 Agent 字典（designer, coder, verifier）
    - device_pool: 设备池（向后兼容）
    - private_worker / worker_manager: Worker 管理
    - conductor_template: Conductor 分析模板
    """
    
    def __init__(self, agents: dict, device_pool, trace, config: dict,
                 private_worker=None, worker_manager=None, backend=None, arch=None):
        """初始化算子工作流
        
        Args:
            agents: Agent 实例字典 (designer, coder, verifier)
            device_pool: 设备池（向后兼容）
            trace: Trace 实例
            config: 配置字典
            private_worker: 私有 Worker 实例
            worker_manager: WorkerManager 实例
            backend: 后端类型（用于 Worker 选择）
            arch: 架构类型（用于 Worker 选择）
        """
        super().__init__(config, trace)
        
        # 算子专用属性
        self.agents = agents
        self.device_pool = device_pool  # 向后兼容
        self.private_worker = private_worker
        self.worker_manager = worker_manager
        self.backend = backend
        self.arch = arch
        
        # 加载 Conductor 模板
        self._load_conductor_template()
    
    def _load_conductor_template(self):
        """加载算子专用的 Conductor 模板"""
        try:
            prompt_dir = get_prompt_path()
            template_path = os.path.join(prompt_dir, "conductor/analyze.j2")
            with open(template_path, 'r', encoding='utf-8') as f:
                self.conductor_template = Template(f.read())
        except Exception as e:
            logger.warning(f"Failed to load conductor template: {e}")
            self.conductor_template = None

