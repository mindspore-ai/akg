# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Default Workflow V2: KernelDesigner → KernelGen ↔ Verifier（基于 Skill 系统）"""

import logging
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.op.langgraph_op.routers import RouterFactory
from akg_agents.core.checker import CodeChecker
from akg_agents.core_v2.workflows.registry import register_workflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class DefaultWorkflowV2(OpBaseWorkflow):
    """Default Workflow V2：基于 Skill 系统的完整 kernel 开发工作流
    
    与旧版 DefaultWorkflow 的区别：
    - 使用 KernelDesigner（动态 Skill 加载）替代 Designer（静态文档加载）
    - 使用 KernelGen（动态 Skill 加载）替代 Coder（静态文档加载）
    
    流程（带可选 CodeChecker）：
    
        kernel_designer -> kernel_gen -> [code_checker] -> verifier -> (失败) -> conductor -> kernel_gen
                                ^                                                               |
                                |_______________________________________________________________+
                                                    (未通过：携带错误信息回到 kernel_gen)
    
    特点：
    - 先设计后编码，适合复杂任务
    - 使用 Skill 系统动态选择知识和策略
    - KernelDesigner 生成的算法草图传递给 KernelGen 指导代码生成
    """
    
    TOOL_NAME = "call_default_workflow_v2"
    
    DESCRIPTION = """
使用 Default Workflow V2 进行完整的 kernel 开发（基于 Skill 系统，包含设计阶段）。

完整流程：
1. KernelDesigner: 基于 Skill 系统分析需求，设计算法方案和优化策略
2. KernelGen: 基于 Skill 系统，根据设计方案生成代码
3. Verifier: 验证正确性和性能
4. Conductor: 分析失败原因并指导修复（如果验证失败）
5. 循环迭代直到成功或达到最大次数

适用场景：
- 需求复杂，需要先进行算法设计和方案规划
- 对性能优化有较高要求，需要设计阶段的优化建议
- 需要完整的设计→开发→验证流程
- 任务涉及复杂算法

注意事项：
- 包含完整的设计、编码、验证流程
- 执行时间较长（通常 2-8 分钟）
- 需要可用的验证环境（GPU/CPU 等设备）
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
                "description": "目标 DSL，如 'triton_cuda', 'triton_ascend', 'cpp', 'cuda_c'"
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
            },
            "previous_code": {
                "type": "string",
                "description": "之前生成的 kernel 代码（可选），用于代码修改/优化场景",
                "default": ""
            },
            "verifier_error": {
                "type": "string",
                "description": "之前 workflow 失败时的 Verifier 错误信息（可选）",
                "default": ""
            },
            "conductor_suggestion": {
                "type": "string",
                "description": "之前 workflow 失败时的 Conductor 修复建议（可选）",
                "default": ""
            },
            "cur_path": {
                "type": "string",
                "description": "自定义工作路径（可选），指定后中间文件存放在 cur_path/logs/，生成的代码存放在 cur_path/code.txt",
                "default": ""
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]
    }
    
    def build_graph(self) -> StateGraph:
        """构建 DefaultWorkflowV2 工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 检查必需的 Agent
        required_agents = ['kernel_designer', 'kernel_gen', 'verifier']
        for agent_name in required_agents:
            if agent_name not in self.agents:
                raise RuntimeError(f"Required agent '{agent_name}' is not available. "
                                 f"Available agents: {list(self.agents.keys())}")
        
        enable_code_checker = self.config.get("enable_code_checker", False)
        
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
        kernel_designer_node = NodeFactory.create_kernel_designer_node(
            self.agents['kernel_designer'],
            self.trace,
            self.config
        )
        kernel_gen_node = NodeFactory.create_kernel_gen_node(
            self.agents['kernel_gen'],
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
            self.conductor_template,
            code_gen_agent="kernel_gen"
        )
        
        # 添加节点
        workflow.add_node("kernel_designer", kernel_designer_node)
        workflow.add_node("kernel_gen", kernel_gen_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)
        
        # 添加边：kernel_designer → kernel_gen
        workflow.add_edge("kernel_designer", "kernel_gen")
        
        if enable_code_checker and code_checker:
            code_checker_node = NodeFactory.create_code_checker_node(
                code_checker,
                self.trace,
                self.config
            )
            workflow.add_node("code_checker", code_checker_node)
            
            workflow.add_edge("kernel_gen", "code_checker")
            
            code_checker_router = RouterFactory.create_code_checker_router(
                self.config,
                max_check_retries=self.config.get("max_code_check_retries", 5),
                code_gen_agent="kernel_gen"
            )
            
            workflow.add_conditional_edges(
                "code_checker",
                code_checker_router,
                {
                    "verifier": "verifier",
                    "kernel_gen": "kernel_gen"
                }
            )
        else:
            workflow.add_edge("kernel_gen", "verifier")
            logger.info("CodeChecker disabled, using direct kernel_gen -> verifier flow")
        
        # 条件边：verifier 后的路由
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
        
        # Conductor 后的路由（指定使用 kernel_gen）
        conductor_router = RouterFactory.create_conductor_router(self.config, code_gen_agent="kernel_gen")
        
        workflow.add_conditional_edges(
            "conductor",
            conductor_router,
            {
                "kernel_gen": "kernel_gen",
                "finish": END
            }
        )
        
        # 设置入口
        workflow.set_entry_point("kernel_designer")
        
        return workflow
    
    @classmethod
    async def ensure_resources(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """确保 workflow 所需的运行时资源已就位"""
        if not workflow_resources.get("private_worker"):
            try:
                from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
                wm = get_worker_manager()
                backend = workflow_resources.get("backend", "cpu")
                arch = workflow_resources.get("arch", "x86_64")
                if not await wm.has_worker(backend=backend, arch=arch):
                    await register_local_worker([0], backend=backend, arch=arch)
                    logger.info(f"[DefaultWorkflowV2] 已注册本地 worker: backend={backend}, arch={arch}")
                if not workflow_resources.get("worker_manager"):
                    workflow_resources["worker_manager"] = wm
            except Exception as e:
                logger.warning(f"[DefaultWorkflowV2] 确保 worker 资源失败: {e}")
    
    @classmethod
    def prepare_config(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """根据 arguments 调整 workflow 配置"""
        super().prepare_config(workflow_resources, arguments)
    
    def format_result(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """格式化 workflow 结果"""
        code = final_state.get("coder_code", "")
        profile_res = final_state.get("profile_res", {})
        verifier_result = final_state.get("verifier_result", False)
        
        status = "success" if verifier_result else "fail"
        
        res = {
            "code": code,
            "profile": str(profile_res) if profile_res else "",
            "status": status,
        }
        
        if not verifier_result:
            verifier_error = final_state.get("verifier_error", "")
            conductor_suggestion = final_state.get("conductor_suggestion", "")
            if verifier_error:
                res["error_information"] = verifier_error
            if conductor_suggestion:
                res["conductor_suggestion"] = conductor_suggestion
        
        cur_path = final_state.get("cur_path", "")
        if cur_path and code and verifier_result:
            code_path = Path(cur_path) / "code.txt"
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")
            logger.info(f"[DefaultWorkflowV2] 代码已保存到: {code_path}")
            res["code_path"] = str(code_path)
        
        return res
