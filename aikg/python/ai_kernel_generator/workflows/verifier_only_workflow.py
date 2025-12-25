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

"""Verifier-only workflow: Verifier → Finish"""

from langgraph.graph import StateGraph, END
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory


class VerifierOnlyWorkflow(BaseWorkflow):
    """Verifier Only Workflow：仅验证已有代码
    
    Flow:
        verifier -> finish
    """
    
    def build_graph(self) -> StateGraph:
        """构建 Verifier-only 工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 创建节点
        verifier_node = NodeFactory.create_verifier_node(
            self.agents['verifier'], 
            self.device_pool, 
            self.trace,
            self.config,
            self.private_worker,
            self.worker_manager,
            self.backend,
            self.arch
        )
        
        # 添加节点
        workflow.add_node("verifier", verifier_node)
        
        # 直接结束
        workflow.add_edge("verifier", END)
        
        # 设置入口
        workflow.set_entry_point("verifier")
        
        return workflow

