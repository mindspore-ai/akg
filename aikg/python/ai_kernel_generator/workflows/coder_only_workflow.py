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
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory
from ai_kernel_generator.utils.langgraph.routers import RouterFactory
from ai_kernel_generator.core.checker import CodeChecker

logger = logging.getLogger(__name__)


class CoderOnlyWorkflow(BaseWorkflow):
    """Coder Only Workflow：跳过设计阶段，直接生成代码
    
    优化后的流程（带 CodeChecker）：
    
        coder -> code_checker -> (通过) -> verifier -> (失败) -> conductor -> coder
                      |                                                       ^
                      +----------------> (未通过) ----------------------------+
                                     (携带错误信息回到 coder)
    
    CodeChecker 的作用：
    - 在 Verifier 之前进行快速的静态代码检查
    - 检测常见的语法错误（如 break、continue、while 等禁止语法）
    - 避免将明显错误的代码送入 Verifier 浪费时间（Verifier 每次执行约 1 分钟）
    """
    
    def build_graph(self) -> StateGraph:
        """构建 Coder-only 工作流图（带 CodeChecker）"""
        workflow = StateGraph(KernelGenState)
        
        # 检查是否启用 CodeChecker（默认禁用，因为 LLM 检查容易产生假阳性）
        enable_code_checker = self.config.get("enable_code_checker", False)
        
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
            self.conductor_template
        )
        
        # 添加节点
        workflow.add_node("coder", coder_node)
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
            
            # 添加边：coder -> code_checker
            workflow.add_edge("coder", "code_checker")
            
            # 条件边：code_checker 后的路由
            code_checker_router = RouterFactory.create_code_checker_router(
                self.config,
                max_check_retries=self.config.get("max_code_check_retries", 5)
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
            # 不启用 CodeChecker，直接 coder -> verifier
            workflow.add_edge("coder", "verifier")
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
        workflow.set_entry_point("coder")
        
        return workflow
