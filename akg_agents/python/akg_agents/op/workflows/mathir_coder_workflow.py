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

"""MathIR + Coder workflow: MathIR -> API recall -> Coder -> CodeChecker -> Verifier."""

import logging

from langgraph.graph import END, StateGraph

from akg_agents.core_v2.workflows.registry import register_workflow
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.op.langgraph_op.routers import RouterFactory
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.utils.code_checker import CodeChecker
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class MathIRCoderWorkflow(OpBaseWorkflow):
    """Run MathIR before the existing coder-only verification loop.

    默认流程（带 CodeChecker）：

        mathIR -> api_recall -> coder -> code_checker -> (通过) -> verifier -> (失败) -> conductor -> coder
                                               |                                                       ^
                                               +----------------> (未通过) ----------------------------+
                                             (携带错误信息回到 coder)

    关闭 CodeChecker 时：

        mathIR -> api_recall -> coder -> verifier -> (成功) -> END
                                               |
                                               +------> (失败) -> conductor -> coder

    MathIR 只在入口执行一次，用于提前生成 expression-level 数学语义；
    API recall 只基于完整 PyTorch task_desc 执行一次，后续 Coder 共享本地落盘结果；
    后续修复循环由 Coder / Verifier / Conductor 完成。
    """

    TOOL_NAME = "use_mathir_coder_workflow"

    DESCRIPTION = """
使用 MathIR + Coder workflow 生成 kernel 代码。

完整流程：
1. MathIR: 从 PyTorch 代码提取 expression-level 数学语义
2. API recall: 基于完整 PyTorch task_desc 召回 Triton API 并持久化到本地 log
3. Coder: 基于任务描述和可选 MathIR/API recall 生成代码
4. CodeChecker: 静态代码检查（默认开启）
5. Verifier: 验证正确性和性能
6. Conductor: 分析失败原因并指导修复
"""

    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {"type": "string", "description": "算子名称"},
            "task_desc": {"type": "string", "description": "任务描述（框架代码）"},
            "dsl": {"type": "string", "description": "目标 DSL"},
            "framework": {"type": "string", "description": "前端框架"},
            "backend": {"type": "string", "description": "目标后端"},
            "arch": {"type": "string", "description": "目标架构"},
            "task_id": {"type": "string", "description": "任务 ID", "default": ""},
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"],
    }

    def build_graph(self) -> StateGraph:
        """构建 MathIR 前置的 Coder-only 工作流图。"""
        workflow = StateGraph(KernelGenState)
        enable_code_checker = self.config.get("enable_code_checker", True)

        if "mathIR" not in self.agents:
            raise ValueError("MathIRCoderWorkflow requires agents['mathIR']")

        code_checker = None
        if enable_code_checker:
            code_checker = CodeChecker(
                backend=self.backend or "",
                dsl=self.agents.get("coder").dsl if self.agents.get("coder") else "",
                config=self.config,
            )
            logger.info("CodeChecker enabled: backend=%s, dsl=%s", self.backend, code_checker.dsl)

        mathIR_node = NodeFactory.create_mathIR_node(
            self.agents["mathIR"],
            self.trace,
            self.config,
        )
        api_recall_node = NodeFactory.create_api_recall_node(
            self.trace,
            self.config,
            self.backend,
        )
        coder_node = NodeFactory.create_coder_node(
            self.agents["coder"],
            self.trace,
        )
        verifier_node = NodeFactory.create_verifier_node(
            self.agents["verifier"],
            self.device_pool,
            self.trace,
            self.config,
            self.private_worker,
            self.worker_manager,
            self.backend,
            self.arch,
        )
        conductor_node = NodeFactory.create_conductor_node(
            self.trace,
            self.config,
            self.conductor_template,
        )

        workflow.add_node("mathIR", mathIR_node)
        workflow.add_node("api_recall", api_recall_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)

        workflow.add_edge("mathIR", "api_recall")
        workflow.add_edge("api_recall", "coder")

        if enable_code_checker and code_checker:
            code_checker_node = NodeFactory.create_code_checker_node(
                code_checker,
                self.trace,
                self.config,
            )
            workflow.add_node("code_checker", code_checker_node)

            codegen_router = RouterFactory.create_codegen_router(
                next_agent="code_checker",
                code_gen_agent="coder",
            )
            workflow.add_conditional_edges(
                "coder",
                codegen_router,
                {
                    "code_checker": "code_checker",
                    "conductor": "conductor",
                },
            )

            code_checker_router = RouterFactory.create_code_checker_router(self.config)
            workflow.add_conditional_edges(
                "code_checker",
                code_checker_router,
                {
                    "verifier": "verifier",
                    "coder": "coder",
                },
            )
        else:
            codegen_router = RouterFactory.create_codegen_router(
                next_agent="verifier",
                code_gen_agent="coder",
            )
            workflow.add_conditional_edges(
                "coder",
                codegen_router,
                {
                    "verifier": "verifier",
                    "conductor": "conductor",
                },
            )
            logger.info("CodeChecker disabled, using direct coder -> verifier flow")

        verifier_router = RouterFactory.create_verifier_router_with_conductor(self.config)
        workflow.add_conditional_edges(
            "verifier",
            verifier_router,
            {
                "conductor": "conductor",
                "finish": END,
            },
        )

        conductor_router = RouterFactory.create_conductor_router(self.config)
        workflow.add_conditional_edges(
            "conductor",
            conductor_router,
            {
                "coder": "coder",
                "finish": END,
            },
        )

        workflow.set_entry_point("mathIR")
        return workflow
