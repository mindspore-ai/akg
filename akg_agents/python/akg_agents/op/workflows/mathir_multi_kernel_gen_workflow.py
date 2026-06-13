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

"""MathIR multi-kernel workflow."""

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
class MathIRMultiKernelGenWorkflow(OpBaseWorkflow):
    """Run MathIR with optional multi-expression sub-kernel generation.

    multi_kernel_gen workflow 流程图：

        mathIR
          |
          v
        api_recall
          |
          |   - 使用完整原始 PyTorch task_desc 做一次 PyTorch/ATen -> Triton API recall
          |   - 写入 api_recall/api_recall_structured.json
          |   - 写入 api_recall/api_recall_rendered.md
          |   - 后续普通 coder 和 multi_expr_coder 子任务共享同一份本地 recall 文本
          |
          +-- multi_kernel_gen=False 或 expression 数量 <= 1
          |       |
          |       v
          |     coder -> code_checker -> verifier
          |        ^                        |
          |        |                        +-- 失败
          |        +------------------------+
          |
          +-- multi_kernel_gen=True 且 expression 数量 > 1
                  |
                  v
              multi_expr_coder
                  |
                  +-- 初始化：
                  |     - 从 mathIR_code["expressions"] 读取所有 expression
                  |     - 为每个 expression 构造独立 sub task
                  |     - 共享 multi_kernel_max_retries 作为所有子 kernel 的总尝试预算
                  |
                  +-- 循环直到所有子 kernel 成功或预算耗尽：
                  |     |
                  |     +-- 选出尚未成功的 expression
                  |     |
                  |     +-- LLM 并发生成/修复：
                  |     |     - 第一次尝试调用 SUB_EXPR_GEN
                  |     |     - 有 verifier_error 后调用 SUB_EXPR_REPAIR
                  |     |     - 禁止 CLI 流式输出，避免并发 token 交错
                  |     |
                  |     +-- 子 CodeChecker 静态检查：
                  |     |     - 检查 Python 语法/import/DSL 合规性
                  |     |     - 失败则把 code_check_errors 写回该 expression，下一轮 repair
                  |     |     - 不占用设备 verify
                  |     |
                  |     +-- Verify 串行执行：
                  |           - 按 expression 顺序逐个运行子验证脚本
                  |           - verify 成功则保存该子 kernel
                  |           - verify 失败则记录错误，等待下一轮 repair
                  |
                  +-- 全部子 kernel verify 成功：
                  |     |
                  |     +-- COMBINE 生成最终 ModelNew
                  |     |
                  |     +-- combined code -> code_checker -> verifier
                  |
                  +-- 任一子 kernel 耗尽重试预算：
                        |
                        +-- 写入 multi_expr_error / verifier_error
                        |
                        +-- workflow finish，任务直接失败

    说明：
    - MathIR 只在入口执行一次。
    - multi_expr_coder 只处理 MathIR 产生多个 expression 的场景。
    - combine 只有在所有子 kernel verify 成功后才会调用。
    - 任一子 kernel 耗尽重试预算时，不进入兜底修复，任务直接失败。
    """

    TOOL_NAME = "use_mathir_multi_kernel_gen_workflow"

    DESCRIPTION = """
使用 MathIR multi-kernel gen workflow 生成 kernel 代码。

完整流程：
1. MathIR: 从 PyTorch 代码提取 expression-level 数学语义
2. API recall: 基于完整 PyTorch task_desc 召回 Triton API 并持久化到本地 log
3. 多 expression 场景: 并行生成/修复子 kernel，串行 verify，全部通过后 combine
4. 单 expression 或关闭 multi_kernel_gen: 走普通 Coder 生成
5. CodeChecker: 静态代码检查（默认开启）
6. Verifier: 验证正确性和性能
7. 失败后直接回到 Coder 修复；子 kernel 耗尽预算时直接失败
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
        """构建 MathIR multi-kernel gen 工作流图。"""
        workflow = StateGraph(KernelGenState)
        enable_code_checker = self.config.get("enable_code_checker", True)

        if "mathIR" not in self.agents:
            raise ValueError("MathIRMultiKernelGenWorkflow requires agents['mathIR']")

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
        multi_expr_coder_node = NodeFactory.create_multi_expr_coder_node(
            self.agents["coder"],
            self.agents["verifier"],
            self.trace,
            self.config,
            self.private_worker,
            self.worker_manager,
            self.backend,
            self.arch,
            code_checker,
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
        workflow.add_node("mathIR", mathIR_node)
        workflow.add_node("api_recall", api_recall_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("multi_expr_coder", multi_expr_coder_node)
        workflow.add_node("verifier", verifier_node)

        workflow.add_edge("mathIR", "api_recall")

        async def route_after_mathIR(state: KernelGenState) -> str:
            expressions = NodeFactory._extract_mathir_expressions(state.get("mathIR_code"))
            multi_kernel_gen = NodeFactory._coerce_bool(
                state.get("multi_kernel_gen", True),
                True,
            )
            if multi_kernel_gen and len(expressions) > 1:
                logger.info(
                    "MathIR produced %s expressions and multi_kernel_gen is enabled; "
                    "routing to multi_expr_coder",
                    len(expressions),
                )
                return "multi_expr_coder"
            return "coder"

        workflow.add_conditional_edges(
            "api_recall",
            route_after_mathIR,
            {
                "multi_expr_coder": "multi_expr_coder",
                "coder": "coder",
            },
        )

        def create_multi_expr_router(next_agent: str):
            async def route_after_multi_expr(state: KernelGenState) -> str:
                if state.get("multi_expr_error"):
                    logger.info("multi_expr_coder exhausted sub-kernel retry budget; finishing as failed")
                    return "finish"
                if state.get("codegen_invalid"):
                    if state.get("step_count", 0) >= self.config.get("max_step", 20):
                        logger.info("multi_expr_coder reached max_step; finishing as failed")
                        return "finish"
                    logger.info("multi_expr_coder produced invalid combined code; routing to coder")
                    return "coder"
                if not state.get("coder_code"):
                    if state.get("step_count", 0) >= self.config.get("max_step", 20):
                        logger.info("multi_expr_coder reached max_step; finishing as failed")
                        return "finish"
                    logger.info("multi_expr_coder did not produce code; routing to coder")
                    return "coder"
                return next_agent

            return route_after_multi_expr

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
                invalid_agent="coder",
                config=self.config,
            )
            workflow.add_conditional_edges(
                "coder",
                codegen_router,
                {
                    "code_checker": "code_checker",
                    "coder": "coder",
                    "finish": END,
                },
            )
            workflow.add_conditional_edges(
                "multi_expr_coder",
                create_multi_expr_router("code_checker"),
                {
                    "code_checker": "code_checker",
                    "coder": "coder",
                    "finish": END,
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
                invalid_agent="coder",
                config=self.config,
            )
            workflow.add_conditional_edges(
                "coder",
                codegen_router,
                {
                    "verifier": "verifier",
                    "coder": "coder",
                    "finish": END,
                },
            )
            workflow.add_conditional_edges(
                "multi_expr_coder",
                create_multi_expr_router("verifier"),
                {
                    "verifier": "verifier",
                    "coder": "coder",
                    "finish": END,
                },
            )
            logger.info("CodeChecker disabled, using direct coder -> verifier flow")

        verifier_router = RouterFactory.create_verifier_router(
            self.config,
            failure_agent="coder",
        )
        workflow.add_conditional_edges(
            "verifier",
            verifier_router,
            {
                "coder": "coder",
                "finish": END,
            },
        )

        workflow.set_entry_point("mathIR")
        return workflow
