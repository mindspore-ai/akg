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

"""OpTaskBuilder Workflow: 多轮交互将用户文字需求转换为KernelBench格式"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderState, OpTaskBuilderStatus
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.core.agent.op_task_builder import OpTaskBuilder

logger = logging.getLogger(__name__)


class OpTaskBuilderWorkflow:
    """OpTaskBuilder Workflow：多轮交互的算子任务构建流程
    
    将用户的自然语言需求转换为标准的KernelBench格式代码。
    支持多轮对话，用于澄清需求和确认生成结果。
    
    Flow:
        user_input -> op_task_builder_agent -> 
          ├─ ready -> END (返回生成的task_desc)
          ├─ need_clarification -> END (返回澄清问题，等待用户补充)
          ├─ need_modification -> END (返回修改建议，等待用户确认)
          └─ unsupported -> END (返回不支持原因)
    
    多轮交互通过多次调用run()方法实现，每次调用处理一轮对话。
    """
    
    def __init__(self, config: dict, trace=None):
        """初始化 OpTaskBuilder Workflow
        
        Args:
            config: 配置字典，需包含agent_model_config等
            trace: Trace实例（可选）
        """
        self.config = config
        self.trace = trace
        self.max_iterations = config.get("op_task_builder_max_iterations", 5)
        self.max_check_retries = config.get("op_task_builder_max_check_retries", 3)
        
        # 创建OpTaskBuilderAgent
        self.op_task_builder_agent = OpTaskBuilder(config)
        
        # 创建节点
        self.op_task_builder_node = NodeFactory.create_op_task_builder_node(
            self.op_task_builder_agent,
            self.trace
        )
    
    def _create_router(self):
        """创建路由函数"""
        async def route_after_op_task_builder(state: OpTaskBuilderState) -> str:
            """根据OpTaskBuilder结果决定下一步
            
            由于是多轮交互，所有状态都结束当前轮次，等待下次用户输入
            """
            status = state.get("status", "")
            
            if status == OpTaskBuilderStatus.READY:
                logger.info("OpTaskBuilder: Status is READY, task_desc generated successfully")
                return "end"
            elif status == OpTaskBuilderStatus.UNSUPPORTED:
                logger.info("OpTaskBuilder: Status is UNSUPPORTED")
                return "end"
            elif status == OpTaskBuilderStatus.NEED_CLARIFICATION:
                logger.info("OpTaskBuilder: Status is NEED_CLARIFICATION")
                return "end"
            elif status == OpTaskBuilderStatus.NEED_MODIFICATION:
                logger.info("OpTaskBuilder: Status is NEED_MODIFICATION")
                return "end"
            else:
                logger.warning(f"OpTaskBuilder: Unknown status '{status}', ending")
                return "end"
        
        return route_after_op_task_builder
    
    def build_graph(self) -> StateGraph:
        """构建 OpTaskBuilder 工作流图
        
        每次调用只执行一轮op_task_builder，然后结束等待下次用户输入
        """
        workflow = StateGraph(OpTaskBuilderState)
        
        # 添加节点
        workflow.add_node("op_task_builder", self.op_task_builder_node)
        
        # 添加条件边
        router = self._create_router()
        workflow.add_conditional_edges(
            "op_task_builder",
            router,
            {
                "end": END
            }
        )
        
        # 设置入口
        workflow.set_entry_point("op_task_builder")
        
        return workflow
    
    def compile(self):
        """编译图
        
        Returns:
            Compiled application
        """
        graph = self.build_graph()
        return graph.compile()
    
    def visualize(self) -> str:
        """生成 Mermaid 流程图
        
        Returns:
            Mermaid diagram as string
        """
        app = self.compile()
        return app.get_graph().draw_mermaid()
    
    async def run(self, 
                  user_input: str,
                  user_feedback: Optional[str] = None,
                  previous_state: Optional[Dict[str, Any]] = None,
                  framework: str = "torch",
                  backend: str = "cuda",
                  arch: str = "a100",
                  dsl: str = "triton") -> Dict[str, Any]:
        """执行一轮OpTaskBuilder交互
        
        Args:
            user_input: 用户的初始需求（第一轮）或新的需求描述
            user_feedback: 用户对上一轮结果的反馈（可选）
                - 当上一轮返回READY状态时，如果用户拒绝，应传入拒绝理由或修改要求
                - 当上一轮返回NEED_CLARIFICATION状态时，应传入用户补充的信息
            previous_state: 上一轮的状态（用于多轮对话）
                - 当用户拒绝READY任务时，previous_state应包含上一轮的完整状态，
                  特别是generated_task_desc字段，以便OpTaskBuilder知道之前生成了什么
            framework: 目标框架
            backend: 目标后端
            arch: 目标架构
            dsl: 目标DSL
            
        Returns:
            Dict包含（详细字段说明见 OpTaskBuilder.run() 的文档）：
                - status: 当前状态（READY/NEED_CLARIFICATION/NEED_MODIFICATION/UNSUPPORTED）
                - generated_task_desc: 生成的代码（status=READY时存在）
                - op_name: 算子名称
                - agent_message: 给用户的消息（所有状态都存在）
                - clarification_question: 澄清问题（status=NEED_CLARIFICATION时存在）
                - modification_suggestion: 修改建议（status=NEED_MODIFICATION时存在）
                - conversation_history: 对话历史（自动累积）
                - iteration: 当前轮次
                - 其他状态字段...
                
        MainOpAgent 使用指南：
            1. READY 状态：
               - 展示 generated_task_desc 给用户确认
               - 用户确认通过：将 generated_task_desc 作为新输入，重新送入 MainOpAgent 进行合法性校验与执行
               - 用户拒绝：调用 workflow.run(user_input=result["user_input"], user_feedback="用户拒绝理由", previous_state=result)
               
            2. NEED_CLARIFICATION 状态：
               - 展示 agent_message（或 clarification_question）给用户，提示用户补充信息
               - agent_message 中包含准确、完整的所有缺失信息列表
               - 用户提供补充信息后，调用 workflow.run(user_input=result["user_input"], user_feedback="用户补充的信息", previous_state=result)
               
            3. NEED_MODIFICATION 状态：
               - 展示 agent_message 和 modification_suggestion 给用户
               - 用户提供反馈后，调用 workflow.run(user_input=result["user_input"], user_feedback="用户反馈", previous_state=result)
               
            4. UNSUPPORTED 状态：
               - 展示 agent_message 给用户，说明不支持的原因
               - 流程结束
                
        使用示例：
            # 第一轮：用户提供初始需求
            result1 = await workflow.run(user_input="实现ReLU算子")
            
            # 如果返回READY，MainOpAgent展示给用户确认
            if result1["status"] == "ready":
                # 用户拒绝，提供反馈
                result2 = await workflow.run(
                    user_input=result1["user_input"],  # 保留原始需求
                    user_feedback="输入shape应该是(32, 1024)而不是(16, 16384)",
                    previous_state=result1  # 保留上一轮状态，包括generated_task_desc
                )
        """
        # 构建初始状态
        if previous_state:
            # 多轮交互：保留上一轮的所有状态信息
            # 包括 generated_task_desc（当用户拒绝READY任务时，需要保留之前生成的代码）
            state = dict(previous_state)
            state["user_feedback"] = user_feedback
            # 保留之前的user_input作为原始需求（如果用户提供了新的user_input，可以在这里更新）
            if user_input and user_input != previous_state.get("user_input", ""):
                # 如果提供了新的user_input，更新它（但通常多轮交互时保留原始需求）
                state["user_input"] = user_input
        else:
            state = {
                "user_input": user_input,
                "user_feedback": user_feedback,
                "framework": framework,
                "backend": backend,
                "arch": arch,
                "dsl": dsl,
                "iteration": 0,
                "max_iterations": self.max_iterations,
                "check_retry_count": 0,
                "max_check_retries": self.max_check_retries,
                "conversation_history": [],
            }
        # session_id 非 TUI 场景可为空；如提供则写入 state 以便日志/流式输出关联
        session_id = str((self.config or {}).get("session_id") or "").strip()
        if session_id:
            state["session_id"] = session_id
        task_label = str((self.config or {}).get("task_label") or "").strip()
        if task_label:
            state["task_label"] = task_label

        # 检查是否超过最大迭代次数
        if state.get("iteration", 0) >= self.max_iterations:
            logger.warning(f"OpTaskBuilder: Reached max iterations ({self.max_iterations})")
            return {
                "status": OpTaskBuilderStatus.UNSUPPORTED,
                "agent_message": f"已达到最大交互次数（{self.max_iterations}次），请重新开始或提供更清晰的需求描述。",
                **state
            }
        
        # 编译并运行workflow
        app = self.compile()
        result = await app.ainvoke(state)
        
        return result
    
    async def run_until_ready(self,
                              user_input: str,
                              get_user_feedback_callback,
                              framework: str = "torch",
                              backend: str = "cuda",
                              arch: str = "a100",
                              dsl: str = "triton") -> Dict[str, Any]:
        """自动多轮交互直到ready或达到上限
        
        Args:
            user_input: 用户的初始需求
            get_user_feedback_callback: 获取用户反馈的回调函数，签名: async (message: str) -> str
            framework: 目标框架
            backend: 目标后端
            arch: 目标架构
            dsl: 目标DSL
            
        Returns:
            最终状态字典
        """
        state = None
        current_input = user_input
        feedback = None
        
        for i in range(self.max_iterations):
            logger.info(f"OpTaskBuilder: Round {i + 1}/{self.max_iterations}")
            
            result = await self.run(
                user_input=current_input,
                user_feedback=feedback,
                previous_state=state,
                framework=framework,
                backend=backend,
                arch=arch,
                dsl=dsl
            )
            
            status = result.get("status", "")
            
            if status == OpTaskBuilderStatus.READY:
                logger.info("OpTaskBuilder: Got READY status, returning result")
                return result
            elif status == OpTaskBuilderStatus.UNSUPPORTED:
                logger.info("OpTaskBuilder: Got UNSUPPORTED status, returning result")
                return result
            elif status in [OpTaskBuilderStatus.NEED_CLARIFICATION, OpTaskBuilderStatus.NEED_MODIFICATION]:
                # 需要用户反馈
                message = result.get("agent_message", "请提供更多信息")
                feedback = await get_user_feedback_callback(message)
                if not feedback:
                    # 用户没有提供反馈，结束
                    return result
                state = result
            else:
                logger.warning(f"OpTaskBuilder: Unknown status '{status}', returning result")
                return result
        
        # 达到最大迭代次数
        logger.warning(f"OpTaskBuilder: Reached max iterations ({self.max_iterations})")
        return result


# 便捷函数：创建并运行单轮OpTaskBuilder
async def run_op_task_builder(user_input: str,
                              config: dict,
                              user_feedback: Optional[str] = None,
                              framework: str = "torch",
                              backend: str = "cuda",
                              arch: str = "a100",
                              dsl: str = "triton") -> Dict[str, Any]:
    """便捷函数：运行单轮OpTaskBuilder
    
    Args:
        user_input: 用户需求
        config: 配置字典
        user_feedback: 用户反馈（可选）
        framework: 目标框架
        backend: 目标后端
        arch: 目标架构
        dsl: 目标DSL
        
    Returns:
        OpTaskBuilder结果字典
    """
    workflow = OpTaskBuilderWorkflow(config)
    return await workflow.run(
        user_input=user_input,
        user_feedback=user_feedback,
        framework=framework,
        backend=backend,
        arch=arch,
        dsl=dsl
    )

