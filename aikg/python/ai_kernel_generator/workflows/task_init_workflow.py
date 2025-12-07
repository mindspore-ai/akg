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

"""TaskInit Workflow: 多轮交互将用户文字需求转换为KernelBench格式"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from ai_kernel_generator.utils.langgraph.task_init_state import TaskInitState, TaskInitStatus
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory
from ai_kernel_generator.core.agent.task_init_agent import TaskInitAgent

logger = logging.getLogger(__name__)


class TaskInitWorkflow:
    """TaskInit Workflow：多轮交互的任务初始化流程
    
    将用户的自然语言需求转换为标准的KernelBench格式代码。
    支持多轮对话，用于澄清需求和确认生成结果。
    
    Flow:
        user_input -> task_init_agent -> 
          ├─ ready -> END (返回生成的task_desc)
          ├─ need_clarification -> END (返回澄清问题，等待用户补充)
          ├─ need_modification -> END (返回修改建议，等待用户确认)
          └─ unsupported -> END (返回不支持原因)
    
    多轮交互通过多次调用run()方法实现，每次调用处理一轮对话。
    """
    
    def __init__(self, config: dict, trace=None):
        """初始化 TaskInit Workflow
        
        Args:
            config: 配置字典，需包含agent_model_config等
            trace: Trace实例（可选）
        """
        self.config = config
        self.trace = trace
        self.max_iterations = config.get("task_init_max_iterations", 5)
        
        # 创建TaskInitAgent
        self.task_init_agent = TaskInitAgent(config)
        
        # 创建节点
        self.task_init_node = NodeFactory.create_task_init_node(
            self.task_init_agent,
            self.trace
        )
    
    def _create_router(self):
        """创建路由函数"""
        async def route_after_task_init(state: TaskInitState) -> str:
            """根据TaskInit结果决定下一步
            
            由于是多轮交互，所有状态都结束当前轮次，等待下次用户输入
            """
            status = state.get("status", "")
            
            if status == TaskInitStatus.READY:
                logger.info("TaskInit: Status is READY, task_desc generated successfully")
                return "end"
            elif status == TaskInitStatus.UNSUPPORTED:
                logger.info("TaskInit: Status is UNSUPPORTED")
                return "end"
            elif status == TaskInitStatus.NEED_CLARIFICATION:
                logger.info("TaskInit: Status is NEED_CLARIFICATION")
                return "end"
            elif status == TaskInitStatus.NEED_MODIFICATION:
                logger.info("TaskInit: Status is NEED_MODIFICATION")
                return "end"
            else:
                logger.warning(f"TaskInit: Unknown status '{status}', ending")
                return "end"
        
        return route_after_task_init
    
    def build_graph(self) -> StateGraph:
        """构建 TaskInit 工作流图
        
        每次调用只执行一轮task_init，然后结束等待下次用户输入
        """
        workflow = StateGraph(TaskInitState)
        
        # 添加节点
        workflow.add_node("task_init", self.task_init_node)
        
        # 添加条件边
        router = self._create_router()
        workflow.add_conditional_edges(
            "task_init",
            router,
            {
                "end": END
            }
        )
        
        # 设置入口
        workflow.set_entry_point("task_init")
        
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
        """执行一轮TaskInit交互
        
        Args:
            user_input: 用户的初始需求
            user_feedback: 用户对上一轮结果的反馈（可选）
            previous_state: 上一轮的状态（用于多轮对话）
            framework: 目标框架
            backend: 目标后端
            arch: 目标架构
            dsl: 目标DSL
            
        Returns:
            Dict包含：
                - status: 当前状态
                - generated_task_desc: 生成的代码（如果ready）
                - op_name: 算子名称
                - agent_message: 给用户的消息
                - 其他状态字段...
        """
        # 构建初始状态
        if previous_state:
            state = dict(previous_state)
            state["user_feedback"] = user_feedback
            # 保留之前的user_input作为原始需求
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
                "conversation_history": [],
            }
        
        # 检查是否超过最大迭代次数
        if state.get("iteration", 0) >= self.max_iterations:
            logger.warning(f"TaskInit: Reached max iterations ({self.max_iterations})")
            return {
                "status": TaskInitStatus.UNSUPPORTED,
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
            logger.info(f"TaskInit: Round {i + 1}/{self.max_iterations}")
            
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
            
            if status == TaskInitStatus.READY:
                logger.info("TaskInit: Got READY status, returning result")
                return result
            elif status == TaskInitStatus.UNSUPPORTED:
                logger.info("TaskInit: Got UNSUPPORTED status, returning result")
                return result
            elif status in [TaskInitStatus.NEED_CLARIFICATION, TaskInitStatus.NEED_MODIFICATION]:
                # 需要用户反馈
                message = result.get("agent_message", "请提供更多信息")
                feedback = await get_user_feedback_callback(message)
                if not feedback:
                    # 用户没有提供反馈，结束
                    return result
                state = result
            else:
                logger.warning(f"TaskInit: Unknown status '{status}', returning result")
                return result
        
        # 达到最大迭代次数
        logger.warning(f"TaskInit: Reached max iterations ({self.max_iterations})")
        return result


# 便捷函数：创建并运行单轮TaskInit
async def run_task_init(user_input: str,
                        config: dict,
                        user_feedback: Optional[str] = None,
                        framework: str = "torch",
                        backend: str = "cuda",
                        arch: str = "a100",
                        dsl: str = "triton") -> Dict[str, Any]:
    """便捷函数：运行单轮TaskInit
    
    Args:
        user_input: 用户需求
        config: 配置字典
        user_feedback: 用户反馈（可选）
        framework: 目标框架
        backend: 目标后端
        arch: 目标架构
        dsl: 目标DSL
        
    Returns:
        TaskInit结果字典
    """
    workflow = TaskInitWorkflow(config)
    return await workflow.run(
        user_input=user_input,
        user_feedback=user_feedback,
        framework=framework,
        backend=backend,
        arch=arch,
        dsl=dsl
    )
