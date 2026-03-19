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

"""Coder-only workflow: Coder → CodeChecker → Verifier"""

import logging
from langgraph.graph import StateGraph, END
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.op.langgraph_op.routers import RouterFactory
from akg_agents.op.utils.code_checker import CodeChecker
from akg_agents.core_v2.workflows.registry import register_workflow

logger = logging.getLogger(__name__)


@register_workflow(scopes=["op"])
class CoderOnlyWorkflow(OpBaseWorkflow):
    """Coder Only Workflow：跳过设计阶段，直接生成代码
    
    优化后的流程（带 CodeChecker + FixCodeGen）：
    
        coder -> code_checker -> (通过) -> verifier -> (失败) -> conductor -+-> coder
                      |                                                     |
                      +-----> (未通过，回到 coder) -----+                    +-> fix_code_gen -> verifier
    
    Conductor 判断逻辑：
    - 局部/小范围错误（如缺少 import、变量名拼写等）→ fix_code_gen（增量修复，使用 fast model）
    - 全局/架构性错误（如算法逻辑重大缺陷）→ coder（完整重新生成）

    CodeChecker 的作用：
    - 在 Verifier 之前进行快速的静态代码检查（ast.parse / py_compile / import 校验）
    - 检测语法错误、编译错误、import 缺失等确定性问题
    - 避免将明显错误的代码送入 Verifier 浪费时间（Verifier 每次执行约 1 分钟）
    """
    
    # ========== 工具配置元数据（用于 KernelAgent 调用）==========
    TOOL_NAME = "use_coder_only_workflow"
    
    DESCRIPTION = """
使用 CoderOnly workflow 生成 kernel 代码（跳过设计阶段）。

完整流程：
1. Coder: 根据任务描述生成代码
2. CodeChecker: 静态代码检查（默认开启）
3. Verifier: 验证正确性和性能
4. Conductor: 分析失败原因并决定修复策略（如果验证失败）
   - 局部错误 → FixCodeGen: 增量修复（search/replace，使用 fast model）
   - 全局错误 → Coder: 完整重新生成
5. 循环迭代直到成功或达到最大次数

适用场景：
- 需求明确，无需额外设计阶段
- 需要完整的代码生成、验证、迭代流程
- 需要自动处理编译错误和性能验证
- 对代码质量要求较高，需要多轮迭代优化

注意事项：
- 此 workflow 会执行完整流程，包括验证和迭代优化
- 执行时间较长（通常 1-5 分钟，取决于验证次数）
- 需要可用的验证环境（GPU/CPU 等设备）
- 如果只需要快速生成代码草稿，建议使用 call_kernel_gen

何时使用：
- 用户明确要求"完整开发"、"验证"、"测试"时
- 用户要求"生成并验证正确性"时
- 任务需要生产级别的代码质量时
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
                "description": "用户额外需求（可选），如性能目标、特殊约束等",
                "default": ""
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend", "arch"]
    }
    
    # ========== Workflow 实现 ==========
    
    def build_graph(self) -> StateGraph:
        """构建 Coder-only 工作流图（带 CodeChecker + FixCodeGen）"""
        workflow = StateGraph(KernelGenState)
        
        # 检查是否启用 CodeChecker（默认禁用，因为 LLM 检查容易产生假阳性）
        enable_code_checker = self.config.get("enable_code_checker", False)
        # FixCodeGen 增量修复（默认启用）
        enable_fix_code_gen = self.config.get("enable_fix_code_gen", True)
        
        # 创建 CodeChecker 实例
        code_checker = None
        if enable_code_checker:
            code_checker = CodeChecker(
                backend=self.backend or "",
                dsl=self.agents.get('coder').dsl if self.agents.get('coder') else "",
                config=self.config
            )
            logger.info(f"CodeChecker enabled: backend={self.backend}, dsl={code_checker.dsl}")
        
        # 创建节点
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
            self.conductor_template,
            enable_fix_code_gen=enable_fix_code_gen,
        )
        
        # 添加节点
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)
        
        # FixCodeGen 节点
        if enable_fix_code_gen:
            fix_code_gen_node = NodeFactory.create_fix_code_gen_node(
                self.trace, self.config
            )
            workflow.add_node("fix_code_gen", fix_code_gen_node)
            workflow.add_edge("fix_code_gen", "verifier")
            logger.info("FixCodeGen enabled: conductor can route to fix_code_gen")
        
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
            
            # 条件边：code_checker 后的路由
            code_checker_router = RouterFactory.create_code_checker_router(
                self.config
            )
            
            workflow.add_conditional_edges(
                "code_checker",
                code_checker_router,
                {
                    "verifier": "verifier",  # 检查通过 → Verifier
                    "coder": "coder"         # 检查失败 → 回到 Coder 修复
                }
            )
        else:
            # 不启用 CodeChecker，直接 coder -> verifier（带 codegen 路由）
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
                "conductor": "conductor",  # 验证失败 → Conductor 分析
                "finish": END              # 验证通过 → 直接结束
            }
        )
        
        # Conductor 后的路由
        conductor_edges = {
            "coder": "coder",
            "finish": END,
        }
        if enable_fix_code_gen:
            conductor_edges["fix_code_gen"] = "fix_code_gen"

        conductor_router = RouterFactory.create_conductor_router(
            self.config,
            enable_fix_code_gen=enable_fix_code_gen,
        )
        
        workflow.add_conditional_edges(
            "conductor",
            conductor_router,
            conductor_edges,
        )
        
        # 设置入口
        workflow.set_entry_point("coder")
        
        return workflow
