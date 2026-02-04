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
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.op.langgraph_op.routers import RouterFactory
from akg_agents.core_v2.workflows.registry import register_workflow


@register_workflow(scopes=["op"])
class ConnectAllWorkflow(OpBaseWorkflow):
    """Connect All Workflow：全连接，所有 Agent 可互相转换
    
    Flow:
        designer ⟷ coder ⟷ verifier
           ↓        ↓        ↓
              finish (智能决策)
    """
    
    # ========== 工具配置元数据（用于 KernelAgent 调用）==========
    TOOL_NAME = "use_connect_all_workflow"
    
    DESCRIPTION = """
使用 ConnectAll workflow 进行灵活的 kernel 开发（智能决策路由）。

特点：
- 全连接图：Designer、Coder、Verifier 之间可以任意跳转
- 智能路由：每个节点执行后，由 AI 决定下一步应该做什么
- 自适应流程：根据当前状态和结果动态调整执行路径

执行流程（示例）：
1. Designer: 设计算法方案
2. 智能路由决策 → Coder: 生成代码
3. 智能路由决策 → Verifier: 验证
4. 智能路由决策 → 如果失败，可能回到 Designer 重新设计，或回到 Coder 修复
5. 循环直到成功

适用场景：
- 需要最大的灵活性，允许流程动态调整
- 任务复杂，可能需要多次设计-编码-验证的循环
- 不确定具体流程，希望 AI 智能决策
- 实验性项目，探索最佳开发路径

与其他 workflow 的区别：
- Default: 固定流程 Designer→Coder→Verifier
- CoderOnly: 固定流程 Coder→Verifier
- ConnectAll: 灵活流程，AI 智能决策每一步

注意事项：
- 执行时间不可预测（通常 2-10 分钟）
- 可能产生较多的迭代次数
- 智能路由依赖 LLM 决策质量
- 适合复杂场景，简单任务建议使用固定流程

何时使用：
- 任务极其复杂，标准流程可能不够用时
- 需要最大灵活性时
- 作为实验和探索使用
- 标准 workflow 失败后的备选方案
"""
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称"
            },
            "task_desc": {
                "type": "string",
                "description": "任务描述（框架代码），必须包含框架实现的完整代码"
            },
            "dsl": {
                "type": "string",
                "description": "目标 DSL，如 'triton', 'cpp', 'cuda_c'"
            },
            "framework": {
                "type": "string",
                "description": "框架，如 'torch', 'mindspore', 'numpy'"
            },
            "backend": {
                "type": "string",
                "description": "后端，如 'cuda', 'cpu', 'npu'"
            },
            "arch": {
                "type": "string",
                "description": "架构，如 'a100', 'x86_64', 'ascend910b4'"
            },
            "task_id": {
                "type": "string",
                "description": "任务 ID（可选）",
                "default": ""
            },
            "user_requirements": {
                "type": "string",
                "description": "用户额外需求（可选）",
                "default": ""
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]
    }
    
    # ========== Workflow 实现 ==========
    
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

