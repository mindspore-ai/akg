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

"""Default workflow: Designer → Coder ↔ Verifier"""

from langgraph.graph import StateGraph, END
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory
from ai_kernel_generator.utils.langgraph.routers import RouterFactory


class DefaultWorkflow(BaseWorkflow):
    """默认 Workflow：Designer → Coder ↔ Verifier
    
    Flow:
        designer -> coder -> verifier
                      ^         |
                      |_________|
                    (if verification fails)
    """
    
    def build_graph(self) -> StateGraph:
        """构建默认工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 检查必需的 Agent
        required_agents = ['designer', 'coder', 'verifier']
        for agent_name in required_agents:
            if agent_name not in self.agents:
                raise RuntimeError(f"Required agent '{agent_name}' is not available. "
                                 f"Available agents: {list(self.agents.keys())}")
        
        # 创建节点
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
        conductor_node = NodeFactory.create_conductor_node(
            self.trace,
            self.config,
            self.conductor_template
        )
        
        # 添加节点
        workflow.add_node("designer", designer_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)  # 新增 Conductor 节点
        
        # 添加边
        workflow.add_edge("designer", "coder")
        workflow.add_edge("coder", "verifier")
        
        # 条件边：verifier 后的路由（验证通过跳过 conductor）
        verifier_router = RouterFactory.create_verifier_router_with_conductor(
            self.config
        )
        
        workflow.add_conditional_edges(
            "verifier",
            verifier_router,
            {
                "conductor": "conductor",  # 验证失败 → Conductor 分析
                "finish": END              # 验证通过 → 直接结束
            }
        )
        
        # Conductor 后的路由
        conductor_router = RouterFactory.create_conductor_router(self.config)
        
        workflow.add_conditional_edges(
            "conductor",
            conductor_router,
            {
                "coder": "coder",
                "finish": END
            }
        )
        
        # 设置入口
        workflow.set_entry_point("designer")
        
        return workflow

