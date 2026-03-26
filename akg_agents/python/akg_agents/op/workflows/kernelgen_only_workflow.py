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

"""KernelGen-only workflow: KernelGen → CodeChecker → Verifier"""

import logging
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.op.langgraph_op.routers import RouterFactory
from akg_agents.op.utils.code_checker import CodeChecker
from akg_agents.core_v2.workflows.registry import register_workflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class KernelGenOnlyWorkflow(OpBaseWorkflow):
    """KernelGen Only Workflow：基于 Skill 系统的内核代码生成工作流
    
    优化后的流程（带 CodeChecker）：
    
        kernel_gen -> code_checker -> (通过) -> verifier -> (失败) -> conductor -> kernel_gen
                          |                                                        ^
                          +----------------> (未通过) -----------------------------+
                                         (携带错误信息回到 kernel_gen)
    
    特点：
    - 使用 KernelGen agent（基于 Skill 系统）直接生成代码
    - 跳过 Designer 阶段，直接生成内核代码
    - 支持动态 Skill 选择和知识注入
    """
    
    # ========== 工具配置元数据（用于 KernelAgent 调用）==========
    TOOL_NAME = "call_kernelgen_workflow"
    
    DESCRIPTION = """
使用 KernelGenOnly workflow 生成 kernel 代码（基于 Skill 系统）。

完整流程：
1. KernelGen: 基于 Skill 系统动态选择知识，生成代码
2. Verifier: 验证正确性和性能
3. Conductor: 分析失败原因并指导修复（如果验证失败）
4. 循环迭代直到成功或达到最大次数

适用场景：
- 需求明确，无需额外设计阶段
- 需要基于特定 DSL 知识（如 Triton Ascend、CPU C++）生成代码
- 需要完整的代码生成、验证、迭代流程
- 对代码质量要求较高，需要多轮迭代优化

注意事项：
- 此 workflow 会执行完整流程，包括验证和迭代优化
- 执行时间较长（通常 1-5 分钟，取决于验证次数）
- 需要可用的验证环境（GPU/CPU 等设备）
- 如果只需要快速生成代码草稿，建议使用 call_kernel_gen
"""
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称，如 'relu', 'matmul', 'softmax', 'layernorm'"
            },
            "task_desc": {
                "type": "string",
                "description": "任务描述（框架代码），必须包含框架实现的完整代码，包括 class Model(nn.Module) 定义"
            },
            "dsl": {
                "type": "string",
                "description": "目标 DSL，如 'triton_cuda'（Triton GPU）, 'triton_ascend'（Triton Ascend）, 'cpp'（C++）"
            },
            "framework": {
                "type": "string",
                "description": "框架，如 'torch'（PyTorch）"
            },
            "backend": {
                "type": "string",
                "description": "后端，如 'cuda'（NVIDIA GPU）, 'cpu'（CPU）, 'ascend'（Ascend NPU）"
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
                "description": "用户额外需求（可选），如性能目标、特殊约束等",
                "default": ""
            },
            "previous_code": {
                "type": "string",
                "description": "之前生成的 kernel 代码（可选），用于代码修改/优化场景。提供后 KernelGen 会基于此代码进行修改而非从零生成",
                "default": ""
            },
            "verifier_error": {
                "type": "string",
                "description": "之前 workflow 失败时的 Verifier 错误信息（可选）。传入后 KernelGen 能看到历史报错并针对性修复",
                "default": ""
            },
            "conductor_suggestion": {
                "type": "string",
                "description": "之前 workflow 失败时的 Conductor 修复建议（可选）。传入后 KernelGen 会参考该建议进行修改",
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
    
    # ========== Workflow 实现 ==========
    
    def build_graph(self) -> StateGraph:
        """构建 KernelGen-only 工作流图（带 CodeChecker）"""
        workflow = StateGraph(KernelGenState)
        
        # CodeChecker 做纯静态检查（语法/编译/import），默认开启
        enable_code_checker = self.config.get("enable_code_checker", True)
        
        # 创建 CodeChecker 实例
        code_checker = None
        if enable_code_checker:
            # KernelGen 没有 dsl 属性，从 verifier 获取
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
            code_gen_agent="kernel_gen"  # 显式指定使用 kernel_gen
        )
        
        # 添加节点
        workflow.add_node("kernel_gen", kernel_gen_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)
        
        if enable_code_checker and code_checker:
            # 创建 CodeChecker 节点
            code_checker_node = NodeFactory.create_code_checker_node(
                code_checker,
                self.trace,
                self.config
            )
            workflow.add_node("code_checker", code_checker_node)
            
            # 代码生成后的路由（处理 max_tokens 截断等异常）
            codegen_router = RouterFactory.create_codegen_router(
                next_agent="code_checker",
                code_gen_agent="kernel_gen"
            )
            workflow.add_conditional_edges(
                "kernel_gen",
                codegen_router,
                {
                    "code_checker": "code_checker",
                    "conductor": "conductor"
                }
            )
            
            # 条件边：code_checker 后的路由（指定使用 kernel_gen）
            code_checker_router = RouterFactory.create_code_checker_router(
                self.config,
                code_gen_agent="kernel_gen"
            )
            
            workflow.add_conditional_edges(
                "code_checker",
                code_checker_router,
                {
                    "verifier": "verifier",       # 检查通过 → Verifier
                    "kernel_gen": "kernel_gen"     # 检查失败 → 回到 KernelGen 修复
                }
            )
        else:
            # 不启用 CodeChecker，直接 kernel_gen -> verifier（带 codegen 路由）
            codegen_router = RouterFactory.create_codegen_router(
                next_agent="verifier",
                code_gen_agent="kernel_gen"
            )
            workflow.add_conditional_edges(
                "kernel_gen",
                codegen_router,
                {
                    "verifier": "verifier",
                    "conductor": "conductor"
                }
            )
            logger.info("CodeChecker disabled, using direct kernel_gen -> verifier flow")
        
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
        workflow.set_entry_point("kernel_gen")
        
        return workflow
    
    # ========== 资源准备与配置 ==========
    
    @classmethod
    async def ensure_resources(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """确保 workflow 所需的运行时资源已就位
        
        如果没有 private_worker，则通过全局 WorkerManager 确保有可用的 worker。
        
        Args:
            workflow_resources: workflow 资源字典
            arguments: 工具调用参数
        """
        if not workflow_resources.get("private_worker"):
            try:
                from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
                wm = get_worker_manager()
                backend = workflow_resources.get("backend", "cpu")
                arch = workflow_resources.get("arch", "x86_64")
                if not await wm.has_worker(backend=backend, arch=arch):
                    await register_local_worker([0], backend=backend, arch=arch)
                    logger.info(f"[KernelGenOnlyWorkflow] 已注册本地 worker: backend={backend}, arch={arch}")
                if not workflow_resources.get("worker_manager"):
                    workflow_resources["worker_manager"] = wm
            except Exception as e:
                logger.warning(f"[KernelGenOnlyWorkflow] 确保 worker 资源失败: {e}")
    
    @classmethod
    def prepare_config(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """根据 arguments 调整 workflow 配置
        
        调用父类 OpBaseWorkflow.prepare_config 完成：
        - log_dir 重定向到 cur_path/logs
        - 创建 WorkflowLogger 并注入 workflow_resources["trace"]
        
        Args:
            workflow_resources: workflow 资源字典（会被就地修改）
            arguments: 工具调用参数
        """
        super().prepare_config(workflow_resources, arguments)
    
    def format_result(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """格式化 workflow 结果
        
        返回标准结果字典：
            - code: 生成的代码
            - profile: 性能分析结果
            - status: success / fail
            - code_path: 代码文件路径（仅在指定 cur_path 且成功时存在）
            - error_information: 失败时的最后一次 verifier 报错
            - conductor_suggestion: 失败时的 conductor 修复建议
        
        Args:
            final_state: workflow 最终状态
            
        Returns:
            格式化的结果字典
        """
        code = final_state.get("coder_code", "")
        profile_res = final_state.get("profile_res", {})
        verifier_result = final_state.get("verifier_result", False)
        
        status = "success" if verifier_result else "fail"
        
        res = {
            "code": code,
            "profile": str(profile_res) if profile_res else "",
            "status": status,
        }
        
        # 失败时附带报错信息和 conductor 建议，供上层（KernelAgent）决策
        if not verifier_result:
            verifier_error = final_state.get("verifier_error", "")
            conductor_suggestion = final_state.get("conductor_suggestion", "")
            if verifier_error:
                res["error_information"] = verifier_error
            if conductor_suggestion:
                res["conductor_suggestion"] = conductor_suggestion
        
        # 如果指定了 cur_path 且验证通过，将代码保存到 cur_path/code.txt
        cur_path = final_state.get("cur_path", "")
        if cur_path and code and verifier_result:
            code_path = Path(cur_path) / "code.txt"
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")
            logger.info(f"[KernelGenOnlyWorkflow] 代码已保存到: {code_path}")
            res["code_path"] = str(code_path)
        
        return res
