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

"""Connect-all workflow: All agents can transition to each other"""

from langgraph.graph import StateGraph, END
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory
from ai_kernel_generator.utils.langgraph.routers import RouterFactory


class ConnectAllWorkflow(BaseWorkflow):
    """Connect All Workflow：全连接，所有 Agent 可互相转换
    
    Flow:
        designer ⟷ coder ⟷ verifier
           ↓        ↓        ↓
              finish (智能决策)
    """
    
    def build_graph(self) -> StateGraph:
        """构建全连接工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 创建所有节点
        designer_node = NodeFactory.create_designer_node(
            self.agents['designer'], 
            self.trace,
            self.config
        )
        coder_node = NodeFactory.create_coder_node(
            self.agents['coder'], 
            self.trace
        )
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
        workflow.add_node("designer", designer_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)
        
        # 创建智能路由器
        smart_router = RouterFactory.create_smart_router(
            self.config,
            self.conductor_template,
            self.config.get("agent_model_config", {})
        )
        
        # 每个节点都可以跳转到其他任意节点（通过智能路由）
        for node_name in ["designer", "coder", "verifier"]:
            workflow.add_conditional_edges(
                node_name,
                smart_router,
                {
                    "designer": "designer",
                    "coder": "coder",
                    "verifier": "verifier",
                    "finish": END
                }
            )
        
        # 设置入口
        workflow.set_entry_point("designer")
        
        return workflow

