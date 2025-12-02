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

"""Base workflow class for LangGraph-based workflows."""

from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.common_utils import get_prompt_path
from jinja2 import Template
import os
import logging

logger = logging.getLogger(__name__)


class BaseWorkflow(ABC):
    """Workflow 基类"""
    
    def __init__(self, agents: dict, device_pool, trace, config: dict,
                 private_worker=None, worker_manager=None, backend=None, arch=None):
        """初始化 Workflow
        
        Args:
            agents: Agent 实例字典 (designer, coder, verifier)
            device_pool: 设备池（向后兼容）
            trace: Trace 实例
            config: 配置字典
            private_worker: 私有 Worker 实例（新增）
            worker_manager: WorkerManager 实例（新增）
            backend: 后端类型（新增，用于 Worker 选择）
            arch: 架构类型（新增，用于 Worker 选择）
        """
        self.agents = agents
        self.device_pool = device_pool  # 向后兼容
        self.private_worker = private_worker
        self.worker_manager = worker_manager
        self.backend = backend
        self.arch = arch
        self.trace = trace
        self.config = config
        self.max_iterations = config.get("max_step", 20)
        
        # 加载 Conductor 模板
        try:
            prompt_dir = get_prompt_path()
            template_path = os.path.join(prompt_dir, "conductor/analyze.j2")
            with open(template_path, 'r', encoding='utf-8') as f:
                self.conductor_template = Template(f.read())
        except Exception as e:
            logger.warning(f"Failed to load conductor template: {e}")
            self.conductor_template = None
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """子类实现具体图结构
        
        Returns:
            StateGraph instance
        """
        pass
    
    def compile(self):
        """编译图
        
        Returns:
            Compiled application
        """
        graph = self.build_graph()
        return graph.compile()
    
    def visualize(self) -> str:
        """生成 Mermaid 流程图
        
        Returns:
            Mermaid diagram as string
        """
        app = self.compile()
        return app.get_graph().draw_mermaid()

