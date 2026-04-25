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
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.op.langgraph_op.routers import RouterFactory
from akg_agents.op.utils.code_checker import CodeChecker
from akg_agents.core_v2.workflows.registry import register_workflow

import logging

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class DefaultWorkflow(OpBaseWorkflow):
    """默认 Workflow：Designer → Coder ↔ Verifier
    
    Flow:
        designer → coder → code_checker → verifier → conductor → coder
                       ↑                                    |
                       +-------- (检查失败，回去修) ---------+
    """
    
    # ========== 工具配置元数据（用于 KernelAgent 调用）==========
    TOOL_NAME = "use_default_workflow"
    
    DESCRIPTION = """
使用 Default workflow 进行完整的 kernel 开发（包含设计阶段）。

完整流程：
1. Designer: 分析需求，设计算法方案和优化策略
2. Coder: 根据设计方案生成代码
3. Verifier: 验证正确性和性能
4. Conductor: 分析失败原因并指导修复（如果验证失败）
5. 循环迭代直到成功或达到最大次数

适用场景：
- 需求复杂，需要先进行算法设计和方案规划
- 对性能优化有较高要求，需要设计阶段的优化建议
- 需要完整的设计→开发→验证流程
- 任务涉及复杂算法（如 attention、LayerNorm 等）

与 CoderOnly workflow 的区别：
- Default: 先设计后编码，更适合复杂任务
- CoderOnly: 直接编码，适合简单明确的任务

注意事项：
- 包含完整的设计、编码、验证流程
- 执行时间较长（通常 2-8 分钟）
- 需要可用的验证环境（GPU/CPU 等设备）
- 设计阶段会提供优化建议和实现策略

何时使用：
- 用户要求"设计并实现"、"优化方案"时
- 任务复杂度高，需要算法设计时
- 对性能有明确优化要求时
"""
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称，如 'attention', 'layernorm', 'matmul', 'softmax'"
            },
            "task_desc": {
                "type": "string",
                "description": "任务描述（框架代码），必须包含框架实现的完整代码，包括 class Model(nn.Module) 定义"
            },
            "dsl": {
                "type": "string",
                "description": "目标 DSL，如 'triton'（Triton GPU）, 'cpp'（C++）, 'cuda_c'（CUDA C）"
            },
            "framework": {
                "type": "string",
                "description": "框架，如 'torch'（PyTorch）, 'mindspore'（MindSpore）, 'numpy'"
            },
            "backend": {
                "type": "string",
                "description": "后端，如 'cuda'（NVIDIA GPU）, 'cpu'（CPU）, 'npu'（Ascend NPU）"
            },
            "arch": {
                "type": "string",
                "description": "架构，如 'a100'（NVIDIA A100）, 'x86_64'（Intel/AMD CPU）, 'ascend910b4'（Ascend 910B）"
            },
            "task_id": {
                "type": "string",
                "description": "任务 ID（可选，用于日志和目录命名）",
                "default": ""
            },
            "user_requirements": {
                "type": "string",
                "description": "用户额外需求（可选），如性能目标、特殊约束、优化重点等",
                "default": ""
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]
    }
    
    # ========== Workflow 实现 ==========
    
    def build_graph(self) -> StateGraph:
        """构建默认工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 检查必需的 Agent
        required_agents = ['designer', 'coder', 'verifier']
        for agent_name in required_agents:
            if agent_name not in self.agents:
                raise RuntimeError(f"Required agent '{agent_name}' is not available. "
                                 f"Available agents: {list(self.agents.keys())}")
        
        # CodeChecker 做纯静态检查（语法/编译/import），默认开启
        enable_code_checker = self.config.get("enable_code_checker", True)
        
        code_checker = None
        if enable_code_checker:
            dsl = ""
            verifier = self.agents.get('verifier')
            if verifier and hasattr(verifier, 'dsl'):
                dsl = verifier.dsl
            code_checker = CodeChecker(
                backend=self.backend or "",
                dsl=dsl,
                config=self.config
            )
            logger.info(f"CodeChecker enabled: backend={self.backend}, dsl={code_checker.dsl}")
        
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
        workflow.add_node("conductor", conductor_node)
        
        # 添加边
        workflow.add_edge("designer", "coder")

        if enable_code_checker and code_checker:
            code_checker_node = NodeFactory.create_code_checker_node(
                code_checker,
                self.trace,
                self.config
            )
            workflow.add_node("code_checker", code_checker_node)

            codegen_router = RouterFactory.create_codegen_router(
                next_agent="code_checker",
                code_gen_agent="coder"
            )
            workflow.add_conditional_edges(
                "coder",
                codegen_router,
                {
                    "code_checker": "code_checker",
                    "conductor": "conductor"
                }
            )

            code_checker_router = RouterFactory.create_code_checker_router(
                self.config,
                code_gen_agent="coder"
            )
            workflow.add_conditional_edges(
                "code_checker",
                code_checker_router,
                {
                    "verifier": "verifier",
                    "coder": "coder"
                }
            )
        else:
            codegen_router = RouterFactory.create_codegen_router(
                next_agent="verifier",
                code_gen_agent="coder"
            )
            workflow.add_conditional_edges(
                "coder",
                codegen_router,
                {
                    "verifier": "verifier",
                    "conductor": "conductor"
                }
            )
            logger.info("CodeChecker disabled, using direct coder -> verifier flow")
        
        # 条件边：verifier 后的路由（验证通过跳过 conductor）
        verifier_router = RouterFactory.create_verifier_router_with_conductor(
            self.config
        )
        
        workflow.add_conditional_edges(
            "verifier",
            verifier_router,
            {
                "conductor": "conductor",
                "finish": END
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
