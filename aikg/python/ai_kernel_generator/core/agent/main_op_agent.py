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

"""
MainOpAgent - 基于 LangGraph 的对话式算子生成系统

"""

import logging
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from ai_kernel_generator.core.agent.op_task_builder import OpTaskBuilder
from ai_kernel_generator.core.sub_agent_registry import get_registry
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.langgraph.conversational_state import ConversationalOpGenState, Message
from ai_kernel_generator.utils.common_utils import ParserFactory

from ai_kernel_generator.utils.main_op_agent_utils import (
    is_operator_related_intent,
    quick_match_sub_agent_preference,
    extract_sub_agent_from_reasoning,
    user_explicitly_requests_evolve,
    user_requests_profile,
    is_modification_request,
    simple_action_heuristic,
    format_agents_info_for_llm
)

from ai_kernel_generator.utils.main_op_agent_display import (
    format_display_message,
    get_hint_message
)

logger = logging.getLogger(__name__)


class MainOpAgent(AgentBase):

    def __init__(self, 
                 config: dict,
                 framework: str = "torch",
                 backend: str = "cuda",
                 arch: str = "a100",
                 dsl: str = "triton"):
        """
        初始化 MainOpAgent
                                 
        """
        context = {
            "agent_name": "main_op_agent",
        }
        # 如果 config 中包含 session_id，添加到 context 中
        # 这样在流式输出启用时，run_llm 方法可以正确获取 session_id
        if "session_id" in config:
            context["session_id"] = config["session_id"]
        super().__init__(context=context, config=config)
        
        self.config = config
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl

        self.model_config = config.get("agent_model_config", {})
        
        # 创建 OpTaskBuilder
        self.op_task_builder = OpTaskBuilder(config=config)
        
        # 设置 OpTaskBuilder 的配置属性
        self.op_task_builder.framework = framework
        self.op_task_builder.backend = backend
        self.op_task_builder.arch = arch
        self.op_task_builder.dsl = dsl
        
        # 意图分类功能
        self.use_intent_classification = config.get("use_intent_classification", True)
        if self.use_intent_classification:
            try:
                # 加载意图分类的 prompt 模板
                self.intent_classification_prompt = self.load_template("main_op_agent/classify_intent.j2")
                # 获取意图分类的解析器
                self.intent_parser = ParserFactory.get_intent_classifier_parser()
                self.intent_format_instructions = self.intent_parser.get_format_instructions()
                logger.info("Intent classification enabled")
            except Exception as e:
                logger.warning(f"Failed to load intent classification resources: {e}, disabling intent classification")
                self.use_intent_classification = False
        else:
            logger.info("Intent classification disabled")
        
        # 子 Agent 选择功能
        self.use_auto_sub_agent_selection = config.get("use_auto_sub_agent_selection", True)
        if self.use_auto_sub_agent_selection:
            try:
                # 加载子 Agent 选择的 prompt 模板
                self.sub_agent_selection_prompt = self.load_template("main_op_agent/select_sub_agent.j2")
                # 获取子 Agent 选择的解析器
                self.sub_agent_parser = ParserFactory.get_sub_agent_selector_parser()
                self.sub_agent_format_instructions = self.sub_agent_parser.get_format_instructions()
                logger.info("Auto sub-agent selection enabled")
            except Exception as e:
                logger.warning(f"Failed to load sub-agent selection resources: {e}, disabling auto selection")
                self.use_auto_sub_agent_selection = False
        else:
            logger.info("Auto sub-agent selection disabled")
        
        # 用户动作分析功能
        self.use_auto_action_analysis = config.get("use_auto_action_analysis", True)
        if self.use_auto_action_analysis:
            try:
                # 加载用户动作分析的 prompt 模板
                self.user_action_analysis_prompt = self.load_template("main_op_agent/analyze_user_action.j2")
                # 获取用户动作分析的解析器
                self.action_analyzer_parser = ParserFactory.get_user_action_analyzer_parser()
                self.action_analyzer_format_instructions = self.action_analyzer_parser.get_format_instructions()
                logger.info("Auto user action analysis enabled")
            except Exception as e:
                logger.warning(f"Failed to load user action analysis resources: {e}, disabling auto analysis")
                self.use_auto_action_analysis = False
        else:
            logger.info("Auto user action analysis disabled")
        

        # 获取子 Agent 注册中心
        self.sub_agent_registry = get_registry()
        
        # 自动发现所有可用的子 Agent
        self.available_workflows = list(self.sub_agent_registry.list_agents().keys())
        logger.info(f"Available sub-agents: {self.available_workflows}")
        
        # 构建对话流程图
        self.app = self._build_conversation_graph()
        
        logger.info("MainOpAgent initialized successfully")

    def _build_conversation_graph(self) -> StateGraph:
        """
        构建基于 LangGraph 的对话流程图

        """
        workflow = StateGraph(ConversationalOpGenState)
        
        # 添加节点
        workflow.add_node("op_task_build", self._op_task_build_node)
        workflow.add_node("user_confirm", self._user_confirm_node)
        workflow.add_node("select_sub_agent", self._select_sub_agent_node)
        workflow.add_node("sub_agent_execution", self._sub_agent_execution_node)
        
        # 设置入口
        workflow.set_entry_point("op_task_build")
        
        # 添加边
        workflow.add_edge("op_task_build", "user_confirm")
        
        # 条件边: 用户确认后的路由
        workflow.add_conditional_edges(
            "user_confirm",
            self._should_generate_code,
            {
                "generate": "select_sub_agent",  # 确认后先选择子 Agent
                "revise": "op_task_build",
                "end": END
            }
        )
        
        # 选择子 Agent 后执行
        workflow.add_edge("select_sub_agent", "sub_agent_execution")
        
        # 子 Agent 执行后：可以重新生成 task、重新调用子 Agent、或结束
        workflow.add_conditional_edges(
            "sub_agent_execution",
            self._should_retry_or_end,
            {
                "retry_task": "op_task_build",  # 重新生成 task code
                "retry_sub_agent": "select_sub_agent",  # 保持 task，重新选择并调用子 Agent
                "end": END
            }
        )
        
        return workflow.compile()

    async def _op_task_build_node(self, state: ConversationalOpGenState) -> Dict[str, Any]:
        """
        OpTaskBuilder 节点: 生成或修改 task 代码
        """
        logger.info("=== OpTaskBuild Node ===")
        
        # 监控对话历史长度
        conversation_history = state.get("conversation_history", [])
        logger.debug(f"Conversation history length: {len(conversation_history)} messages")
        
        # 如果对话历史接近上限，发出警告
        from ai_kernel_generator.utils.langgraph.conversational_state import MAX_CONVERSATION_HISTORY_LENGTH
        if len(conversation_history) >= MAX_CONVERSATION_HISTORY_LENGTH * 0.8:
            logger.warning(f"⚠️ Conversation history is getting long: {len(conversation_history)}/{MAX_CONVERSATION_HISTORY_LENGTH} messages. "
                          f"Older messages will be automatically dropped.")
        
        user_request = state.get("user_request", "")
        conversation_history = state.get("conversation_history", [])
        is_first_turn = len(conversation_history) <= 1  # 只有用户的第一条消息
        has_previous_code = bool(state.get("task_code"))
        is_modification_req = is_modification_request(user_request, has_previous_code)
        
        if self.use_intent_classification and is_first_turn and not is_modification_req:
            conversation_history = state.get("conversation_history", [])
            
            try:
                logger.info(f"Running intent classification for: {user_request[:50]}...")
                
                # 构建对话历史上下文
                history_context = ""
                if conversation_history:
                    history_context = "\n".join([
                        f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
                        for msg in conversation_history[-3:]  # 只保留最近3轮
                    ])
                
                # 构建 prompt 输入
                input_data = {
                    "user_input": user_request,
                    "conversation_history": history_context,
                    "format_instructions": self.intent_format_instructions,
                }
                
                # 调用 LLM 进行意图分类
                model_name = self.model_config.get("intent_classifier", "default")
                llm_result, prompt, reasoning = await self.run_llm(
                    self.intent_classification_prompt,
                    input_data,
                    model_name
                )
                
                # 解析 LLM 输出
                try:
                    parsed = ParserFactory.robust_parse(llm_result, self.intent_parser)
                    intent = getattr(parsed, 'intent', 'unclear')
                    message = getattr(parsed, 'message', '')
                    confidence = float(getattr(parsed, 'confidence', 0.5))
                except Exception as parse_error:
                    logger.warning(f"Failed to parse intent classification result: {parse_error}")
                    # 解析失败，默认允许继续
                    intent = "unclear"
                    message = ""
                    confidence = 0.3
                
                # 判断是否是算子相关
                is_operator_related = is_operator_related_intent(intent, confidence, threshold=0.6)
                
                logger.info(f"Intent classification result: intent={intent}, confidence={confidence:.2f}, operator_related={is_operator_related}")
                
                # 记录意图信息（用于调试和日志）
                state["last_intent"] = intent
                state["last_intent_confidence"] = confidence
                
                # 如果不是算子相关，直接返回拒绝消息
                if not is_operator_related:
                    logger.info("Non-operator request detected, returning rejection message")
                    
                    # 根据是否是第一轮对话，显示不同的拒绝消息
                    if is_first_turn:
                        # 第一轮对话：显示详细的引导消息
                        rejection_msg = f"""⚠️ 很抱歉，我无法处理您的请求。

{message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 我是 AI Kernel 算子开发助手，专门帮助您：

1. 生成高性能算子代码（如 ReLU、MatMul、LayerNorm、Softmax 等）
2. 优化现有算子的性能
3. 提供算子开发的技术建议

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 如果您有算子开发需求，请告诉我：

• 算子类型（例如：ReLU 激活函数、矩阵乘法、Layer Normalization 等）
• 输入 tensor 的 shape 和数据类型（例如：(batch_size, channels, height, width), dtype=float32）
• 算子的参数要求（例如：Softmax 的 dim 维度、LayerNorm 的 normalized_shape 等）
• 目标硬件平台（例如：CUDA GPU、Ascend NPU 等）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

示例请求：
"生成 ReLU 激活函数，输入 shape 是 (1024, 1024)"
"实现矩阵乘法，A 是 (m, k)，B 是 (k, n)"
"""
                    else:
                        # 多轮对话中：显示简短的提示
                        rejection_msg = "⚠️ 您的输入似乎与当前算子开发任务无关。请按照上述任务继续生成！"

                    return {
                        "task_code": "",
                        "op_name": "",
                        "op_description": rejection_msg,
                        "current_step": "rejected_by_intent",
                        "user_confirmed": False,
                        "user_feedback": None
                    }
            
            except Exception as e:
                # 意图分类失败，记录日志但继续执行
                logger.warning(f"Intent classification failed: {e}, continuing to OpTaskBuilder")
                # 继续执行，让 OpTaskBuilder 处理
        
        # 如果用户已确认当前的 task_code，跳过重新生成
        user_confirmed = state.get("user_confirmed", False)
        has_task_code = bool(state.get("task_code"))
        logger.info(f"OpTaskBuild check: user_confirmed={user_confirmed}, has_task_code={has_task_code}")
        
        if user_confirmed and has_task_code:
            logger.info("✓ User confirmed existing task code, skipping rebuild")
            return {
                "current_step": "proceed_to_generation"
            }
        
        user_request = state.get("user_request", "")
        conversation_history = state.get("conversation_history", [])
        previous_task_code = state.get("task_code")
        user_feedback = state.get("user_feedback")
        
        try:
            # 构建 OpTaskBuilderState（从 ConversationalOpGenState 中提取）
            from ai_kernel_generator.utils.langgraph.op_task_builder_state import OpTaskBuilderState
            
            op_task_builder_state: OpTaskBuilderState = {
                "user_input": user_request,
                "user_feedback": user_feedback,
                "conversation_history": conversation_history,
                "generated_task_desc": previous_task_code,
                "framework": state.get("framework", "torch"),
                "backend": state.get("backend", "cuda"),
                "arch": state.get("arch", "a100"),
                "dsl": state.get("dsl", "triton"),
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 10),
            }
            
            # 直接调用 OpTaskBuilder.run()，传递完整的 state
            result = await self.op_task_builder.run(op_task_builder_state)
            
            # 从 result 中提取信息
            from ai_kernel_generator.utils.langgraph.op_task_builder_state import OpTaskBuilderStatus
            
            status = result.get("status", OpTaskBuilderStatus.NEED_CLARIFICATION)
            task_code = result.get("generated_task_desc", "")
            op_name = result.get("op_name", "custom_op")
            description = result.get("agent_message", "")
            reasoning = result.get("agent_reasoning", "")
            
            # 处理 OpTaskBuilder 的不同状态
            if status == OpTaskBuilderStatus.NEED_CLARIFICATION:
                # 需要澄清：返回澄清消息
                logger.info("OpTaskBuilder needs clarification")
                clarification_msg = f"⚠️ 需要更多信息：\n\n{description}\n\n请提供更详细的说明。"
                
                new_message = Message(
                    role="assistant",
                    content=clarification_msg,
                    timestamp=datetime.now().isoformat()
                )
                
                return {
                    "task_code": "",
                    "op_name": op_name,
                    "op_description": clarification_msg,
                    "task_reasoning": reasoning,
                    "task_init_status": status,  # 添加这个字段
                    "conversation_history": [new_message],
                    "current_step": "user_confirm",
                    "user_feedback": None,
                    "user_confirmed": False
                }
            
            elif status == OpTaskBuilderStatus.UNSUPPORTED:
                # 不支持的需求
                logger.info("OpTaskBuilder: unsupported request")
                
                new_message = Message(
                    role="assistant",
                    content=description,
                    timestamp=datetime.now().isoformat()
                )
                
                return {
                    "task_code": "",
                    "op_name": op_name,
                    "op_description": description,
                    "task_reasoning": reasoning,
                    "task_init_status": status,  # 添加这个字段
                    "conversation_history": [new_message],
                    "current_step": "unsupported",
                    "user_feedback": None,
                    "user_confirmed": False
                }
            
            else:
                # READY 或其他状态：正常返回
                logger.info(f"OpTaskBuilder returned status: {status}")

                # 添加 assistant 消息到历史
                new_message = Message(
                    role="assistant",
                    content=f"Generated task code for '{op_name}':\n{description}",
                    timestamp=datetime.now().isoformat()
                )

                result_state = {
                    "task_code": task_code,
                    "op_name": op_name,
                    "op_description": description,
                    "task_reasoning": reasoning,
                    "task_init_status": status,  # 添加这个字段
                    "conversation_history": [new_message],
                    "current_step": "user_confirm",
                    # 清除 user_feedback 和 user_confirmed，防止 LangGraph 自动循环
                    "user_feedback": None,
                    "user_confirmed": False
                }
                logger.info(f"Returning state with task_init_status: {result_state.get('task_init_status')}")
                return result_state

        except Exception as e:
            logger.error(f"OpTaskBuild node failed: {e}")
            return {
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1,
                "should_continue": False
            }

    async def _user_confirm_node(self, state: ConversationalOpGenState) -> Dict[str, Any]:
        """
        用户确认节点: 标记等待用户确认状态
        
        如果用户在第一轮就明确要求"生成并测试性能"，自动确认继续执行
        """
        logger.info("=== User Confirm Node ===")
        
        task_code = state.get("task_code", "")
        op_name = state.get("op_name", "")
        user_request = state.get("user_request", "")
        
        # 检查用户是否明确要求"生成并测试性能"
        # 如果是，自动确认，不需要用户再次确认
        if user_requests_profile(user_request):
            logger.info(f"User requested generate + profile in initial request, auto-confirming")
            return {
                "current_step": "user_confirm",
                "user_confirmed": True  # 自动确认
            }
        
        logger.info(f"Task code generated for op: {op_name}, waiting for user confirmation")
        
        return {
            "current_step": "waiting_for_user_confirmation"
        }

    def _should_generate_code(self, state: ConversationalOpGenState) -> str:
        """
        决定下一步: 生成代码 / 修订 task / 结束
        """
        user_confirmed = state.get("user_confirmed", False)
        user_feedback = state.get("user_feedback")
        
        if user_confirmed:
            return "generate"
        elif user_feedback:
            return "revise"
        else:
            return "end"
    
    def _should_retry_or_end(self, state: ConversationalOpGenState) -> str:
        """
        决定子 Agent 执行后的下一步: 重新生成 task / 重新调用子 Agent / 结束
        """
        retry_requested = state.get("retry_requested", False)
        retry_sub_agent_only = state.get("retry_sub_agent_only", False)
        
        if retry_requested:
            logger.info("User requested retry task, returning to op_task_build")
            return "retry_task"
        elif retry_sub_agent_only:
            logger.info("User requested retry sub-agent only, returning to select_sub_agent")
            return "retry_sub_agent"
        else:
            logger.info("No retry requested, ending workflow")
            return "end"

    async def _select_sub_agent_node(self, state: ConversationalOpGenState) -> Dict[str, Any]:
        """
        子 Agent 选择节点: 使用 LLM 选择最合适的子 Agent
        """
        logger.info("=== Select Sub Agent Node ===")
        
        # 如果用户已经指定了 sub_workflow，跳过自动选择
        if state.get("sub_workflow_specified_by_user"):
            specified_sub_agent = state.get("sub_workflow", "codeonly")
            logger.info(f"✅ User has explicitly specified sub-agent: {specified_sub_agent}")
            logger.info(f"   Skipping LLM selection, using user's choice directly")
            return {
                "sub_workflow": specified_sub_agent,
                "sub_agent_selection_reasoning": f"用户明确要求使用 {specified_sub_agent} 子Agent",
                "sub_agent_selection_confidence": 1.0
            }
        
        # 如果禁用了自动选择，使用默认的 codeonly
        if not self.use_auto_sub_agent_selection:
            logger.info("Auto sub-agent selection disabled, using default: codeonly")
            return {
                "sub_workflow": "codeonly",
                "sub_agent_selection_reasoning": "Auto selection disabled, using default"
            }
        
        task_code = state.get("task_code", "")
        op_name = state.get("op_name", "")
        op_description = state.get("op_description", "")
        user_request = state.get("user_request", "")
        
        # 🔍 首先检查用户是否明确要求使用 evolve
        conversation_history = state.get("conversation_history", [])
        if user_explicitly_requests_evolve(user_request, conversation_history):
            logger.info("User explicitly requested evolve, using evolve sub-agent")
            return {
                "sub_workflow": "evolve",
                "sub_agent_selection_reasoning": "用户明确要求使用 evolve 进行性能优化",
                "sub_agent_selection_confidence": 1.0
            }
        
        try:
            # 获取所有子 Agent 的详细信息
            agents_info = self.sub_agent_registry.get_agents_detailed_info()
            
            # 格式化子 Agent 信息供 LLM 选择
            agents_info_text = format_agents_info_for_llm(agents_info)
            
            # 构建 prompt 输入
            input_data = {
                "task_code": task_code,
                "op_name": op_name,
                "op_description": op_description,
                "user_request": user_request,
                "available_agents": agents_info_text,
                "has_generated_code": bool(state.get("generated_code")),  # 是否已有生成的代码
                "format_instructions": self.sub_agent_format_instructions,
            }
            
            # 调用 LLM 选择子 Agent
            model_name = self.model_config.get("sub_agent_selector", "default")
            llm_result, prompt, reasoning = await self.run_llm(
                self.sub_agent_selection_prompt,
                input_data,
                model_name
            )
            
            # 解析 LLM 输出
            try:
                parsed = ParserFactory.robust_parse(llm_result, self.sub_agent_parser)
                selected_agent = getattr(parsed, 'selected_agent', 'codeonly')
                selection_reasoning = getattr(parsed, 'reasoning', '')
                confidence = float(getattr(parsed, 'confidence', 0.5))
            except Exception as parse_error:
                logger.warning(f"Failed to parse sub-agent selection result: {parse_error}, using default")
                selected_agent = "codeonly"
                selection_reasoning = "Parse error, using default"
                confidence = 0.0
            
            # 验证选择的子 Agent 是否存在
            if not self.sub_agent_registry.is_registered(selected_agent):
                logger.warning(f"Selected agent '{selected_agent}' not registered, using default: codeonly")
                selected_agent = "codeonly"
                selection_reasoning += " (fallback to default due to invalid selection)"
            
            logger.info(f"Selected sub-agent: {selected_agent} (confidence: {confidence:.2f})")
            logger.info(f"Selection reasoning: {selection_reasoning}")
            
            return {
                "sub_workflow": selected_agent,
                "sub_agent_selection_reasoning": selection_reasoning,
                "sub_agent_selection_confidence": confidence
            }
            
        except Exception as e:
            logger.exception(f"Sub-agent selection failed: {e}, using default: codeonly")
            return {
                "sub_workflow": "codeonly",
                "sub_agent_selection_reasoning": f"Selection failed: {str(e)}, using default",
                "sub_agent_selection_confidence": 0.0
            }

    async def _sub_agent_execution_node(self, state: ConversationalOpGenState) -> Dict[str, Any]:
        """
        子 Agent 执行节点: 通过注册中心调用子 Agent 生成算子代码
        """
        logger.info("=== Sub Agent Execution Node ===")
        
        task_code = state.get("task_code", "")
        op_name = state.get("op_name", "")
        task_id = state.get("task_id", "default_task")
        sub_workflow = state.get("sub_workflow", "codeonly")
        
        try:
            # 从注册中心获取子 Agent
            sub_agent = self.sub_agent_registry.get_agent(
                agent_name=sub_workflow,
                config=self.config,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                dsl=self.dsl
            )
            
            if sub_agent is None:
                raise ValueError(f"Sub-agent '{sub_workflow}' not found in registry")
            
            agent_info = sub_agent.get_detailed_info()
            logger.info(f"Using sub-agent: {sub_agent.get_name()} - {agent_info.get('description', '')}")
            
            # 准备子 Agent 执行参数
            execute_kwargs = {
                "generated_code": state.get("generated_code", ""),  # 传递已生成的代码
                "device_id": state.get("device_id", 0)
            }
            
            # 如果是 codeonly 子 Agent，根据用户请求判断 task_type
            if sub_workflow == "codeonly":
                # 优先使用当前轮的用户输入，如果没有则使用初始请求
                # 这样可以支持多轮对话中切换 task_type
                current_input = state.get("current_user_input", "")
                user_request = state.get("user_request", "")
                check_input = current_input if current_input else user_request
                
                if user_requests_profile(check_input):
                    execute_kwargs["task_type"] = "profile"
                    logger.info(f"User requested performance testing (check_input: '{check_input[:50]}...'), setting task_type='profile'")
                else:
                    execute_kwargs["task_type"] = "precision_only"
                    logger.info(f"Standard code generation (check_input: '{check_input[:50]}...'), setting task_type='precision_only'")
            
            success, result = await sub_agent.execute(
                task_code=task_code,
                op_name=op_name,
                task_id=task_id,
                **execute_kwargs
            )
            
            # 提取结果
            generated_code = result.get("generated_code", "")
            verifier_result = result.get("verification_result", False)
            verifier_error = result.get("verification_error", "")
            profile_res = result.get("profile_result")
            
            # 添加结果消息到历史
            if success:
                result_msg = f"✓ Successfully generated {op_name} using {sub_workflow} sub-agent"
            else:
                result_msg = f"✗ Failed to generate {op_name}: {verifier_error}"
            
            new_message = Message(
                role="assistant",
                content=result_msg,
                timestamp=datetime.now().isoformat()
            )
            
            return {
                "generated_code": generated_code,
                "generation_success": success,
                "verification_result": verifier_result,
                "verification_error": verifier_error,
                "profile_result": profile_res,
                "conversation_history": [new_message],
                "current_step": "completed",
                "retry_requested": False,  # 重置标志，防止无限循环
                "retry_sub_agent_only": False  # 重置标志，防止无限循环
            }
            
        except Exception as e:
            logger.error(f"Sub agent execution failed: {e}")
            error_message = Message(
                role="assistant",
                content=f"✗ Sub agent execution failed: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
            return {
                "generation_success": False,
                "generation_error": str(e),
                "conversation_history": [error_message],
                "current_step": "failed",
                "retry_requested": False,  # 重置标志，防止无限循环
                "retry_sub_agent_only": False  # 重置标志，防止无限循环
            }

    async def _analyze_user_action(self, state: Dict[str, Any], user_input: str) -> tuple:
        """
        使用 LLM 分析用户的动作意图

        Args:
            state: 当前状态
            user_input: 用户输入

        Returns:
            tuple: (action, is_new_operator)
            - action: 建议的操作 ('confirm', 'revise', 'retry', 'retry_sub_agent', 'cancel')
            - is_new_operator: 是否是新算子需求 (bool)
        """
        if not self.use_auto_action_analysis:
            return simple_action_heuristic(state, user_input), False

        try:
            # 构建对话历史文本
            conversation_history = state.get("conversation_history", [])
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
                    for msg in conversation_history[-10:]  # 保留最近10轮
                ])

            # 构建 prompt 输入
            input_data = {
                "user_input": user_input,
                "conversation_history": history_text,
                "has_task_code": bool(state.get("task_code")),
                "has_generated_code": bool(state.get("generated_code")),
                "op_name": state.get("op_name", ""),
                "format_instructions": self.action_analyzer_format_instructions,
            }

            # 调用 LLM 分析用户动作
            model_name = self.model_config.get("action_analyzer", "default")
            llm_result, prompt, reasoning = await self.run_llm(
                self.user_action_analysis_prompt,
                input_data,
                model_name
            )

            # 解析 LLM 输出
            try:
                parsed = ParserFactory.robust_parse(llm_result, self.action_analyzer_parser)
                suggested_action = getattr(parsed, 'suggested_action', 'revise')
                analysis_reasoning = getattr(parsed, 'reasoning', '')
                confidence = float(getattr(parsed, 'confidence', 0.5))
                is_new_operator = getattr(parsed, 'is_new_operator', False)
            except Exception as parse_error:
                logger.warning(f"Failed to parse action analysis result: {parse_error}")
                # 解析失败，使用启发式规则
                return simple_action_heuristic(state, user_input), False

            logger.info(f"LLM suggested action: {suggested_action} (confidence: {confidence:.2f})")
            logger.debug(f"Analysis reasoning: {analysis_reasoning}")
            
            if is_new_operator:
                logger.info(f"🆕 LLM detected NEW OPERATOR request (is_new_operator=True)")

            # 保存分析推理到状态中（用于后续处理无关问题）
            state["last_action_reasoning"] = analysis_reasoning

            # 验证建议的 action 是否合法
            valid_actions = ['confirm', 'revise', 'retry', 'retry_sub_agent', 'cancel']
            if suggested_action not in valid_actions:
                logger.warning(f"Invalid suggested action: {suggested_action}, using revise")
                return 'revise', False

            return suggested_action, is_new_operator

        except Exception as e:
            logger.error(f"Action analysis failed: {e}, using heuristic")
            return simple_action_heuristic(state, user_input), False

    async def start_conversation(self, user_request: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        开始新的对话
        
        Args:
            user_request: 用户的初始请求
            task_id: 任务 ID（可选）
            
        Returns:
            初始状态，包含生成的 task 代码等信息
        """
        import hashlib
        import time
        
        if task_id is None:
            task_id = hashlib.md5(f"{user_request}_{time.time()}".encode()).hexdigest()[:8]
        
        # 初始化状态
        initial_state = {
            "user_request": user_request,
            "framework": self.framework,
            "backend": self.backend,
            "arch": self.arch,
            "dsl": self.dsl,
            "task_id": task_id,
            "config": self.config,
            "conversation_history": [Message(
                role="user",
                content=user_request,
                timestamp=datetime.now().isoformat()
            )],
            "iteration": 0,
            "max_iterations": 10,
            "error_count": 0,
            "user_confirmed": False,
            "should_continue": True,
            "available_workflows": self.available_workflows,
            "sub_workflow": "codeonly",  # 默认使用 codeonly
        }
        
        # 执行到用户确认节点
        result = await self.app.ainvoke(initial_state, {
            "recursion_limit": 100
        })

        logger.info(f"start_conversation result has task_init_status: {result.get('task_init_status')}")
        
        # 添加显示消息和提示消息
        result["display_message"] = format_display_message(result)
        result["hint_message"] = get_hint_message(result)
        
        return result

    async def continue_conversation(self, 
                                   current_state: Dict[str, Any],
                                   user_input: str,
                                   action: str = "auto") -> Dict[str, Any]:
        """
        继续对话（修改 task 代码、确认生成、或重试）
        
        Args:
            current_state: 当前状态
            user_input: 用户输入
            action: 用户动作
                - 'auto': 自动使用 LLM 分析用户意图
                - 'revise': 修改 task 代码
                - 'confirm': 确认并生成代码
                - 'retry': 重新生成 task 代码
                - 'retry_sub_agent': 重新生成 Triton 代码
                - 'cancel': 取消对话
            
        Returns:
            更新后的状态
        """
        # 添加用户消息到历史
        user_message = Message(
            role="user",
            content=user_input,
            timestamp=datetime.now().isoformat()
        )
        
        if "conversation_history" not in current_state:
            current_state["conversation_history"] = []
        current_state["conversation_history"].append(user_message)
        
        # 保存当前轮的用户输入（用于多轮对话中判断task_type等）
        current_state["current_user_input"] = user_input
        
        quick_matched_sub_agent = quick_match_sub_agent_preference(user_input)
        if quick_matched_sub_agent:
            logger.info(f"⚡ Quick match: User requested '{quick_matched_sub_agent}' sub-agent")
            current_state["sub_workflow"] = quick_matched_sub_agent
            current_state["sub_workflow_specified_by_user"] = True
            
            # 根据当前状态决定action
            # 判断是否已经执行过 sub_agent
            has_attempted_generation = (
                current_state.get("generated_code") or  # 有生成的代码
                current_state.get("generation_success") is not None or
                current_state.get("verification_result") is not None or  # 有验证结果标记
                "sub_agent" in current_state.get("current_step", "")  # 当前在sub_agent阶段
            )
            
            if has_attempted_generation:
                action = "retry_sub_agent"
                logger.info("Quick match: Setting action to 'retry_sub_agent' (has attempted generation)")
            elif current_state.get("task_code"):
                action = "confirm"
                logger.info("Quick match: Setting action to 'confirm' (has task_code, no generation yet)")
        
        # 如果 action 是 'auto'，使用 LLM 自动分析用户意图
        is_new_operator = False  # 默认不是新算子
        if action == "auto":
            action, is_new_operator = await self._analyze_user_action(current_state, user_input)
            logger.info(f"Auto-analyzed action: {action}, is_new_operator: {is_new_operator}")
            
            # 检查LLM推理中是否提到子Agent
            if action == "retry_sub_agent" and not current_state.get("sub_workflow_specified_by_user"):
                reasoning = current_state.get("last_action_reasoning", "")
                llm_detected_sub_agent = extract_sub_agent_from_reasoning(reasoning)
                if llm_detected_sub_agent:
                    logger.info(f"LLM detected: User wants '{llm_detected_sub_agent}' sub-agent")
                    current_state["sub_workflow"] = llm_detected_sub_agent
                    current_state["sub_workflow_specified_by_user"] = True

        if action == "confirm":
            # 用户确认，继续生成代码
            current_state["user_confirmed"] = True
            current_state["user_feedback"] = None
            current_state["retry_requested"] = False
            current_state["retry_sub_agent_only"] = False
        elif action == "revise":
            # 用户要求修改 task code
            current_state["user_confirmed"] = False
            current_state["user_feedback"] = user_input
            current_state["user_request"] = user_input  # 更新请求
            current_state["retry_requested"] = False
            current_state["retry_sub_agent_only"] = False
        elif action == "retry":
            # 用户要求在子 Agent 执行后重新生成 task code
            # 如果是新算子需求，需要清空更多状态
            if is_new_operator:
                logger.info("=" * 80)
                logger.info("🆕 NEW OPERATOR REQUEST DETECTED (is_new_operator=True)")
                logger.info(f"   Current operator: '{current_state.get('op_name', 'None')}'")
                logger.info(f"   New request: '{user_input[:80]}...'")
                logger.info("   Clearing all previous state and restarting...")
                logger.info("=" * 80)
                
                # 更新 user_request 为新需求
                current_state["user_request"] = user_input
                current_state["user_feedback"] = user_input
                
                # 清空所有旧状态（包括 task_code 和 op_name）
                current_state["task_code"] = ""
                current_state["op_name"] = ""
                current_state["op_description"] = ""
                current_state["task_reasoning"] = ""
                current_state["task_init_status"] = None
                
                # 清空生成的代码和验证结果
                current_state["generated_code"] = ""
                current_state["generation_success"] = False
                current_state["verification_result"] = False
                current_state["verification_error"] = ""
                current_state["profile_result"] = None
                
                # 重置标志
                current_state["retry_requested"] = True
                current_state["retry_sub_agent_only"] = False
                current_state["user_confirmed"] = False
                current_state["sub_workflow_specified_by_user"] = False
                current_state["sub_workflow"] = "codeonly"  # 重置为默认
            else:
                # 普通的 retry（不是新算子）
                current_state["retry_requested"] = True
                current_state["retry_sub_agent_only"] = False
                current_state["user_confirmed"] = False
                current_state["user_feedback"] = user_input if user_input else None
                current_state["user_request"] = user_input if user_input else current_state.get("user_request", "")
                # 清空之前生成的代码，因为要重新生成 task
                current_state["generated_code"] = ""
                current_state["generation_success"] = False
                current_state["verification_result"] = False
                current_state["verification_error"] = ""
                current_state["profile_result"] = None
        elif action == "retry_sub_agent":
            # 用户要求只重新调用子 Agent，保持当前 task code
            current_state["retry_sub_agent_only"] = True
            current_state["retry_requested"] = False
            
            # 关键修改：只有切换到 codeonly/evolve 时才清空代码
            # kernel_verifier 需要保留已生成的代码进行性能分析
            target_sub_agent = current_state.get("sub_workflow", "codeonly")
            if target_sub_agent != "kernel_verifier":
                # 清除之前生成的代码，让工作流可以重新执行 sub_agent
                logger.info(f"Clearing generated_code for sub-agent: {target_sub_agent}")
                current_state["generated_code"] = ""
                current_state["generation_success"] = False
                current_state["verification_result"] = False
                current_state["verification_error"] = ""
                current_state["profile_result"] = None
            else:
                # kernel_verifier 需要保留代码进行性能测试
                logger.info("Keeping generated_code for kernel_verifier (performance analysis)")
            
            # 跳过 op_task_build 的重新生成，直接进入 select_sub_agent
            current_state["user_confirmed"] = True
            current_state["user_feedback"] = None
            # 保持原有的 user_request，不更新
            
            # 安全检查：确保有 task_code
            if not current_state.get("task_code"):
                logger.error("❌ Cannot retry sub-agent: No task_code available!")
                logger.error("This should not happen. Please check the workflow.")
                # 回退到重新生成 task
                current_state["retry_requested"] = True
                current_state["retry_sub_agent_only"] = False
                current_state["user_confirmed"] = False
            else:
                logger.info("=" * 80)
                logger.info("🔄 Retry sub-agent mode activated:")
                logger.info(f"  - sub_workflow: {current_state.get('sub_workflow')}")
                logger.info(f"  - sub_workflow_specified_by_user: {current_state.get('sub_workflow_specified_by_user')}")
                logger.info(f"  - user_confirmed: True (skip op_task_build rebuild)")
                logger.info(f"  - has task_code: {bool(current_state.get('task_code'))}")
                logger.info("=" * 80)
        elif action == "cancel":  # cancel
            # 判断是正常退出还是无关问题
            reasoning = current_state.get("last_action_reasoning", "")
            is_irrelevant = "无关" in reasoning or "不相关" in reasoning or "irrelevant" in reasoning.lower()

            if is_irrelevant:
                # 用户输入与算子开发无关，但继续对话
                logger.info("User input is irrelevant to operator development, continue conversation")
                current_state["current_step"] = "irrelevant_input"
                current_state["op_description"] = """⚠️ 抱歉，您的输入似乎与算子开发无关。

我是 AI Kernel 算子开发助手，专门帮助您：
• 生成高性能算子代码（如 ReLU、MatMul、LayerNorm 等）
• 优化现有算子的性能
• 提供算子开发的技术建议

如果您有算子开发需求，请告诉我具体的算子类型和参数要求。"""
                current_state["should_continue"] = True  # 继续对话
            else:
                # 正常退出
                logger.info("User requested to end conversation")
                current_state["current_step"] = "cancelled"
                current_state["should_continue"] = False  # 退出对话

            current_state["retry_requested"] = False
            current_state["retry_sub_agent_only"] = False
            return current_state
        else:
            # 未知的 action，默认使用 revise
            logger.warning(f"Unknown action: {action}, using revise")
            current_state["user_confirmed"] = False
            current_state["user_feedback"] = user_input
            current_state["user_request"] = user_input
            current_state["retry_requested"] = False
            current_state["retry_sub_agent_only"] = False

        # 继续执行流程
        result = await self.app.ainvoke(current_state, {
            "recursion_limit": 100
        })
        
        # 添加显示消息和提示消息
        result["display_message"] = format_display_message(result)
        result["hint_message"] = get_hint_message(result)
        
        return result

    def save_conversation(self, state: Dict[str, Any], filepath: str):
        """保存对话历史到文件"""
        import json
        
        conversation_data = {
            "task_id": state.get("task_id"),
            "op_name": state.get("op_name"),
            "conversation_history": state.get("conversation_history", []),
            "task_code": state.get("task_code"),
            "generated_code": state.get("generated_code"),
            "success": state.get("generation_success", False)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation saved to: {filepath}")
