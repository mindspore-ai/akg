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

"""OpTaskBuilder: 将用户文字需求转换为评测任务格式的Agent"""

import logging
import ast
from typing import Tuple, Dict, Any, Optional

from akg_agents.core_v2.agents import AgentBase, register_agent
from akg_agents.utils.common_utils import ParserFactory
from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderState, OpTaskBuilderStatus
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.core.worker.manager import get_worker_manager, register_local_worker
from akg_agents.op.config.config_validator import load_config

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class OpTaskBuilder(AgentBase):
    """
    将用户文字需求转换为评测任务格式的 Agent
    
    负责理解用户需求并生成符合评测任务规范（如 KernelBench 或 SOL-ExecBench）的任务描述代码。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_op_task_builder"
    DESCRIPTION = """
将用户的文字需求转换为评测任务格式的任务描述代码（task_desc）。

功能：
- 理解用户需求并提取关键信息
- 生成符合评测任务规范的代码（支持 KernelBench 或 SOL-ExecBench 格式）
- 自动进行静态和运行时验证
- 支持多轮交互和需求澄清

适用场景：
- 用户只提供了文字需求描述，没有提供 task_desc 代码
- 需要将"生成 relu 算子"这样的描述转换为标准格式代码
- 需要多轮对话澄清需求细节（如 shape、dtype 等）

⚠️ 注意：
- 如果用户已经提供了符合格式的 task_desc 代码，无需调用此工具
- task_desc 是后续代码生成和验证的必要输入
- 生成的 task_desc 需要请用户确认后再进行下一步

输出：评测任务格式的 task_desc 代码
"""
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
                    "user_input": {
                        "type": "string",
                "description": "用户的需求描述（必须包含完整的用户原始需求，不要省略任何细节）"
                    },
                    "framework": {
                        "type": "string",
                        "description": "目标框架（例如：'torch', 'mindspore'）",
                        "default": "torch"
                    },
                    "backend": {
                        "type": "string",
                        "description": "目标硬件后端（例如：'cuda', 'ascend'）",
                        "default": "cuda"
                    },
                    "arch": {
                        "type": "string",
                        "description": "目标硬件架构（例如：'a100', 'ascend910b4'）",
                        "default": "a100"
                    },
                    "dsl": {
                        "type": "string",
                        "description": "目标 DSL（例如：'triton_cuda', 'triton_ascend'）",
                        "default": "triton"
                    },
                    "bench_type": {
                        "type": "string",
                        "description": "基准测试类型（例如：'kernelbench', 'sol'）",
                        "default": "kernelbench"
                    },
                    "user_feedback": {
                        "type": "string",
                        "description": "用户反馈（可选，用于多轮交互）",
                        "default": ""
                    }
                },
                "required": ["user_input"]
    }
    
    def __init__(self):
        """初始化OpTaskBuilder"""
        context = {
            "agent_name": "op_task_builder",
            "task_label": "main",
        }
        super().__init__(context=context)
        
        # 加载prompt模板
        self.op_task_builder_prompt = self.load_template("op_task_builder/build_op_task.j2")
        
        # 获取解析器
        self.parser = ParserFactory.get_op_task_builder_parser()
        self.format_instructions = self.parser.get_format_instructions()
        
        # 交互计数
        self.step_count = 0
        
        # 从配置读取最大检查重试次数（包括静态和运行时检查）
        self.max_check_retries = 3
        
    
    def _check_task_desc_static(self, code: str) -> Tuple[bool, str]:
        """静态检查task_desc代码是否符合KernelBench规范
        
        Args:
            code: 生成的task_desc代码
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        try:
            tree = ast.parse(code)
            
            has_model_class = False
            has_forward_method = False
            has_get_inputs = False
            has_get_init_inputs = False
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'Model':
                    has_model_class = True
                    # 检查是否有forward方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                            has_forward_method = True
                elif isinstance(node, ast.FunctionDef):
                    if node.name == 'get_inputs':
                        has_get_inputs = True
                    elif node.name == 'get_init_inputs':
                        has_get_init_inputs = True
            
            missing = []
            if not has_model_class:
                missing.append("class Model")
            elif not has_forward_method:
                missing.append("Model.forward() method")
            if not has_get_inputs:
                missing.append("function get_inputs()")
            if not has_get_init_inputs:
                missing.append("function get_init_inputs()")
            
            if missing:
                return False, f"Missing required components: {', '.join(missing)}"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error in generated code: {str(e)}"
        except Exception as e:
            return False, f"Error parsing generated code: {str(e)}"
    
    async def _check_task_desc_runtime(
        self, 
        task_desc: str, 
        op_name: str,
        framework: str,
        backend: str,
        arch: str,
        dsl: str,
        config: Any,
        timeout: int = 60
    ) -> Tuple[bool, str]:
        """运行时检查 task_desc 代码是否能正确执行
        
        Args:
            task_desc: task_desc 代码字符串
            op_name: 算子名称
            framework: 框架类型
            backend: 后端类型
            arch: 硬件架构
            dsl: DSL 类型
            config: 配置对象
            timeout: 超时时间（秒）
            
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        
                
        try:
            # 检查是否已有匹配的 worker，如果没有则注册本地 worker
            has_worker = await get_worker_manager().has_worker(backend=backend, arch=arch)
            if not has_worker:
                try:
                    await register_local_worker([0], backend=backend, arch=arch)
                    logger.info(f"Registered local worker for backend={backend}, arch={arch}")
                except Exception as e:
                    logger.warning(f"Failed to register local worker: {e}")
                    # 继续执行，可能有其他 worker 可用
            
            # 选择一个可用的 worker
            worker = await get_worker_manager().select(backend=backend, arch=arch)
            
            if not worker:
                logger.error(f"No worker available for backend={backend}, arch={arch}")
                return False, f"No worker available for backend={backend}, arch={arch}"
            
            # 创建 KernelVerifier（使用默认 task_id="0"）
            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_desc,
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=config,
                worker=worker
            )
            
            # 执行运行时检查
            passed, error = await verifier.check_task_desc_runtime(task_desc, timeout)
            return passed, error
        except Exception as e:
            logger.warning(f"Runtime check failed with exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, f"Runtime check exception: {str(e)}"
    
    def _build_conversation_context(self, state: OpTaskBuilderState) -> str:
        """构建对话历史上下文
        
        Args:
            state: 当前状态
            
        Returns:
            格式化的对话历史字符串
        """
        history = state.get("conversation_history", [])
        if not history:
            return ""
        
        context_parts = []
        for item in history[-5:]:  # 只保留最近5轮对话
            role = item.get("role", "unknown")
            content = item.get("content", "")
            context_parts.append(f"[{role}]: {content}")
        
        return "\n".join(context_parts)
    
    def _prepare_llm_input(self, state: OpTaskBuilderState) -> Dict[str, Any]:
        """准备 LLM 输入数据
        
        Args:
            state: 当前状态
            
        Returns:
            prompt 输入数据字典
        """
        user_input = state.get("user_input", "")
        user_feedback = state.get("user_feedback", "")
        conversation_context = self._build_conversation_context(state)
        static_check_error = state.get("static_check_error", "")
        runtime_check_error = state.get("runtime_check_error", "")
        previous_task_desc = state.get("generated_task_desc", "")
        
        return {
            "user_input": user_input,
            "user_feedback": user_feedback,
            "conversation_history": conversation_context,
            "static_check_error": static_check_error,
            "runtime_check_error": runtime_check_error,
            "previous_task_desc": previous_task_desc,
            "format_instructions": self.format_instructions,
            "framework": state.get("framework", ""),
            "backend": state.get("backend", ""),
            "bench_type": state.get("bench_type", "kernelbench"),
        }
    
    async def _call_llm_and_parse(self, input_data: Dict[str, Any], model_level: str = "standard") -> Dict[str, Any]:
        """调用 LLM 并解析输出
        
        Args:
            input_data: prompt 输入数据
            model_level: 模型级别
        Returns:
            字典，包含以下字段：
                - parsed: 解析后的对象（解析成功时存在）
                - op_name: 算子名称
                - status: 状态
                - task_desc: 生成的代码
                - message: LLM 返回的消息
                - llm_reasoning: LLM 推理过程
                - prompt: prompt 字符串
                - parse_error: 解析错误（解析失败时存在，成功时为 None）
        """
        # 更新context用于日志
        self.context.update({
            "hash": f"OpTaskBuilder@{self.step_count}",
            "step": self.step_count,
        })
        
        # 调用LLM
        llm_result, prompt, reasoning = await self.run_llm(
            self.op_task_builder_prompt, 
            input_data, 
            model_level or "standard"
        )
        logger.debug(f"RAW LLM RESULT: {llm_result}")
        
        # 解析LLM输出
        try:
            parsed = ParserFactory.robust_parse(llm_result, self.parser)
            
            op_name = getattr(parsed, 'op_name', None)
            status = getattr(parsed, 'status', OpTaskBuilderStatus.NEED_CLARIFICATION)
            task_desc = getattr(parsed, 'task_code', "")
            message = getattr(parsed, 'description', "")
            llm_reasoning = getattr(parsed, 'reasoning', reasoning)
            
            return {
                "op_name": op_name,
                "status": status,
                "task_desc": task_desc,
                "message": message,
                "llm_reasoning": llm_reasoning,
                "prompt": prompt,
                "parse_error": None
            }
        except Exception as parse_error:
            return {
                "op_name": None,
                "status": None,
                "task_desc": None,
                "message": None,
                "llm_reasoning": None,
                "prompt": prompt,
                "parse_error": parse_error
            }
    
    def _handle_parse_error(self, 
                           parse_error: Exception,
                           retry_count: int,
                           max_retries: int,
                           history_entry: Dict[str, str],
                           prompt: str,
                           state: OpTaskBuilderState) -> Tuple[Optional[Dict[str, Any]], bool]:
        """处理解析错误
        
        Args:
            parse_error: 解析异常
            retry_count: 当前重试次数
            max_retries: 最大重试次数
            history_entry: 对话历史条目
            prompt: prompt 字符串
            state: 当前状态
            
        Returns:
            Tuple[result_dict, should_continue]
            - result_dict: 如果达到最大重试次数，返回错误状态字典；否则为 None
            - should_continue: 是否应该继续重试
        """
        logger.warning(f"OpTaskBuilder: Parse failed (attempt {retry_count}/{max_retries}): {parse_error}")
        
        if retry_count >= max_retries:
            # 达到最大重试次数，返回 NEED_MODIFICATION
            logger.error(f"OpTaskBuilder: Reached max check retries ({max_retries}), returning NEED_MODIFICATION")
            current_round_history = [history_entry, {
                "role": "assistant",
                "content": f"代码生成失败（已重试{retry_count}次）：无法解析LLM输出"
            }]
            
            return {
                "status": OpTaskBuilderStatus.NEED_MODIFICATION,
                "generated_task_desc": "",
                "op_name": "unknown",
                "agent_message": f"代码生成过程中出现错误（已自动重试{retry_count}次）：无法解析LLM输出。请重试或提供更清晰的需求描述。",
                "modification_suggestion": f"解析错误: {str(parse_error)}",
                "agent_reasoning": f"Parse error: {str(parse_error)}",
                "static_check_passed": False,
                "static_check_error": "",
                "runtime_check_passed": True,
                "runtime_check_error": "",
                "check_retry_count": 0,  # 重置检查重试计数
                "max_check_retries": max_retries,
                "op_task_builder_prompt": prompt,
                "conversation_history": current_round_history,
            }, False
        else:
            # 可以重试，继续循环（不更新错误信息，因为解析失败不需要加到prompt）
            # check_retry_count 已在循环开始时更新，这里不需要再更新
            logger.info(f"OpTaskBuilder: Retrying after parse failure (attempt {retry_count}/{max_retries})")
            return None, True
    
    def _handle_static_check_failure(self,
                                    static_check_error: str,
                                    task_desc: str,
                                    op_name: str,
                                    llm_reasoning: str,
                                    retry_count: int,
                                    max_retries: int,
                                    history_entry: Dict[str, str],
                                    prompt: str,
                                    state: OpTaskBuilderState) -> Tuple[Optional[Dict[str, Any]], bool]:
        """处理静态检查失败
        
        Args:
            static_check_error: 静态检查错误信息
            task_desc: 生成的代码
            op_name: 算子名称
            llm_reasoning: LLM 推理过程
            retry_count: 当前重试次数（已在循环开始时 +1）
            max_retries: 最大重试次数
            history_entry: 对话历史条目
            prompt: prompt 字符串
            state: 当前状态
            
        Returns:
            Tuple[result_dict, should_continue]
        """
        logger.warning(f"OpTaskBuilder: Static check failed (attempt {retry_count}/{max_retries}): {static_check_error}")
        
        if retry_count >= max_retries:
            # 达到最大重试次数，返回 NEED_MODIFICATION
            logger.error(f"OpTaskBuilder: Reached max check retries ({max_retries}), returning NEED_MODIFICATION")
            # 获取累积的错误信息（用于 agent_message 和 modification_suggestion）
            accumulated_static_error = state.get("static_check_error", "")
            # 如果有累积的错误，使用累积的错误；否则使用当前错误
            final_static_error = accumulated_static_error if accumulated_static_error else static_check_error
            current_round_history = [history_entry, {
                "role": "assistant",
                "content": f"生成的代码格式检查未通过（已重试{retry_count}次）: {static_check_error}"
            }]
            
            return {
                "status": OpTaskBuilderStatus.NEED_MODIFICATION,
                "generated_task_desc": task_desc,
                "op_name": op_name,
                "agent_message": f"生成的代码存在格式问题（已自动重试{retry_count}次）：{static_check_error}\n请确认您的需求，我将重新生成。",
                "modification_suggestion": final_static_error,  # 使用累积的错误信息
                "agent_reasoning": llm_reasoning,
                "static_check_passed": False,
                "static_check_error": "",  # 重置错误信息，不影响下次用户交互
                "runtime_check_passed": True,
                "runtime_check_error": "",  # 重置错误信息，不影响下次用户交互
                "check_retry_count": 0,  # 重置检查重试计数
                "max_check_retries": max_retries,
                "op_task_builder_prompt": prompt,
                "conversation_history": current_round_history,
            }, False
        else:
            # 可以重试，更新状态并继续循环
            logger.info(f"OpTaskBuilder: Retrying after static check failure (attempt {retry_count}/{max_retries})")
            # 累积静态检查错误（追加而不是覆盖）
            existing_static_error = state.get("static_check_error", "")
            if existing_static_error:
                state["static_check_error"] = f"{existing_static_error}\n\n[重试 {retry_count}] {static_check_error}"
            else:
                state["static_check_error"] = f"[重试 {retry_count}] {static_check_error}"
            state["runtime_check_error"] = ""  # 清除之前的运行时检查错误（静态检查失败时，运行时检查还未执行）
            state["generated_task_desc"] = task_desc  # 保留之前的代码，供 prompt 参考
            # check_retry_count 已在循环开始时更新，这里不需要再更新
            return None, True
    
    def _handle_runtime_check_failure(self,
                                     runtime_check_error: str,
                                     task_desc: str,
                                     op_name: str,
                                     llm_reasoning: str,
                                     retry_count: int,
                                     max_retries: int,
                                     history_entry: Dict[str, str],
                                     prompt: str,
                                     state: OpTaskBuilderState) -> Tuple[Optional[Dict[str, Any]], bool]:
        """处理运行时检查失败
        
        Args:
            runtime_check_error: 运行时检查错误信息
            task_desc: 生成的代码
            op_name: 算子名称
            llm_reasoning: LLM 推理过程
            retry_count: 当前重试次数（已在循环开始时 +1）
            max_retries: 最大重试次数
            history_entry: 对话历史条目
            prompt: prompt 字符串
            state: 当前状态
            
        Returns:
            Tuple[result_dict, should_continue]
        """
        logger.warning(f"OpTaskBuilder: Runtime check failed (attempt {retry_count}/{max_retries}): {runtime_check_error}")
        
        if retry_count >= max_retries:
            # 达到最大重试次数，返回 NEED_MODIFICATION
            logger.error(f"OpTaskBuilder: Reached max check retries ({max_retries}), returning NEED_MODIFICATION")
            # 获取累积的错误信息（用于 agent_message 和 modification_suggestion）
            accumulated_runtime_error = state.get("runtime_check_error", "")
            # 如果有累积的错误，使用累积的错误；否则使用当前错误
            final_runtime_error = accumulated_runtime_error if accumulated_runtime_error else runtime_check_error
            current_round_history = [history_entry, {
                "role": "assistant",
                "content": f"生成的代码运行时检查未通过（已重试{retry_count}次）: {runtime_check_error}"
            }]
            
            return {
                "status": OpTaskBuilderStatus.NEED_MODIFICATION,
                "generated_task_desc": task_desc,
                "op_name": op_name,
                "agent_message": f"生成的代码运行时检查失败（已自动重试{retry_count}次）：{runtime_check_error}\n请确认您的需求，我将重新生成。",
                "modification_suggestion": final_runtime_error,  # 使用累积的错误信息
                "agent_reasoning": llm_reasoning,
                "static_check_passed": True,
                "static_check_error": "",  # 重置错误信息，不影响下次用户交互
                "runtime_check_passed": False,
                "runtime_check_error": "",  # 重置错误信息，不影响下次用户交互
                "check_retry_count": 0,  # 重置检查重试计数
                "max_check_retries": max_retries,
                "op_task_builder_prompt": prompt,
                "conversation_history": current_round_history,
            }, False
        else:
            # 可以重试，更新状态并继续循环
            logger.info(f"OpTaskBuilder: Retrying after runtime check failure (attempt {retry_count}/{max_retries})")
            state["static_check_error"] = ""  # 静态检查已通过，清除错误
            # 累积运行时检查错误（追加而不是覆盖）
            existing_runtime_error = state.get("runtime_check_error", "")
            if existing_runtime_error:
                state["runtime_check_error"] = f"{existing_runtime_error}\n\n[重试 {retry_count}] {runtime_check_error}"
            else:
                state["runtime_check_error"] = f"[重试 {retry_count}] {runtime_check_error}"
            state["generated_task_desc"] = task_desc  # 保留之前的代码，供 prompt 参考
            # check_retry_count 已在循环开始时更新，这里不需要再更新
            return None, True
    
    def _build_ready_state(self,
                          task_desc: str,
                          op_name: str,
                          message: str,
                          llm_reasoning: str,
                          runtime_check_passed: bool,
                          history_entry: Dict[str, str],
                          prompt: str,
                          max_retries: int,
                          state: OpTaskBuilderState) -> Dict[str, Any]:
        """构建 READY 状态的返回字典
        
        Args:
            task_desc: 生成的代码
            op_name: 算子名称
            message: LLM 返回的消息
            llm_reasoning: LLM 推理过程
            runtime_check_passed: 运行时检查是否通过
            history_entry: 对话历史条目
            prompt: prompt 字符串
            max_retries: 最大重试次数
            state: 当前状态
            
        Returns:
            READY 状态字典
        """
        check_summary = "格式验证通过，运行时验证通过"
        logger.info(f"OpTaskBuilder: Generated valid op_task_desc for op '{op_name}' ({check_summary})")
        current_round_history = [history_entry, {
            "role": "assistant", 
            "content": f"已生成 {op_name} 算子代码，{check_summary}。"
        }]
        
        return {
            "status": OpTaskBuilderStatus.READY,
            "generated_task_desc": task_desc,
            "op_name": op_name,
            "agent_message": message or f"已成功生成 {op_name} 算子的KernelBench格式代码。",
            "agent_reasoning": llm_reasoning,
            "static_check_passed": True,
            "static_check_error": "",
            "runtime_check_passed": runtime_check_passed,
            "runtime_check_error": "",
            "check_retry_count": 0,  # 重置重试计数
            "max_check_retries": max_retries,
            "op_task_builder_prompt": prompt,
            "conversation_history": current_round_history,
        }
    
    def _build_unsupported_state(self,
                                 message: str,
                                 llm_reasoning: str,
                                 history_entry: Dict[str, str],
                                 prompt: str,
                                 max_retries: int) -> Dict[str, Any]:
        """构建 UNSUPPORTED 状态的返回字典
        
        Args:
            message: LLM 返回的消息
            llm_reasoning: LLM 推理过程
            history_entry: 对话历史条目
            prompt: prompt 字符串
            max_retries: 最大重试次数
            
        Returns:
            UNSUPPORTED 状态字典
        """
        current_round_history = [history_entry, {
            "role": "assistant",
            "content": message or "该需求不在AIKG支持范围内。"
        }]
        
        return {
            "status": OpTaskBuilderStatus.UNSUPPORTED,
            "agent_message": message or "抱歉，当前系统不支持该类型的需求。AIKG主要用于生成高性能算子kernel代码。",
            "agent_reasoning": llm_reasoning,
            "check_retry_count": 0,  # 重置重试计数
            "max_check_retries": max_retries,
            "runtime_check_passed": True,
            "runtime_check_error": "",
            "op_task_builder_prompt": prompt,
            "iteration": 0,  # 会在调用处更新
            "conversation_history": current_round_history,
        }
    
    def _build_clarification_state(self,
                                  message: str,
                                  op_name: Optional[str],
                                  llm_reasoning: str,
                                  history_entry: Dict[str, str],
                                  prompt: str,
                                  max_retries: int) -> Dict[str, Any]:
        """构建 NEED_CLARIFICATION 状态的返回字典
        
        Args:
            message: LLM 返回的消息
            op_name: 算子名称（可选）
            llm_reasoning: LLM 推理过程
            history_entry: 对话历史条目
            prompt: prompt 字符串
            max_retries: 最大重试次数
            
        Returns:
            NEED_CLARIFICATION 状态字典
        """
        current_round_history = [history_entry, {
            "role": "assistant",
            "content": message or "需要更多信息。"
        }]
        
        return {
            "status": OpTaskBuilderStatus.NEED_CLARIFICATION,
            "clarification_question": message,  # 用于调用方 Agent 展示给用户
            "agent_message": message or "请提供更多信息以便生成算子代码。",  # 给用户的消息
            "agent_reasoning": llm_reasoning,
            "op_name": op_name,
            "check_retry_count": 0,  # 重置重试计数
            "max_check_retries": max_retries,
            "runtime_check_passed": True,
            "runtime_check_error": "",
            "op_task_builder_prompt": prompt,
            "iteration": 0,  # 会在调用处更新
            "conversation_history": current_round_history,
        }
    
    async def _handle_ready_status(self,
                                  task_desc: str,
                                  op_name: str,
                                  message: str,
                                  llm_reasoning: str,
                                  prompt: str,
                                  retry_count: int,
                                  max_retries: int,
                                  history_entry: Dict[str, str],
                                  state: OpTaskBuilderState) -> Tuple[Optional[Dict[str, Any]], bool]:
        """处理 READY 状态（执行静态检查和运行时检查）
        
        Args:
            task_desc: 生成的代码
            op_name: 算子名称
            message: LLM 返回的消息
            llm_reasoning: LLM 推理过程
            prompt: prompt 字符串
            retry_count: 当前重试次数
            max_retries: 最大重试次数
            history_entry: 对话历史条目
            state: 当前状态
            
        Returns:
            Tuple[result_dict, should_continue]
        """
        # 获取配置参数（默认值为空，由 check_task_config 在 LangGraphTask 中验证）
        dsl = state.get("dsl", "")
        backend = state.get("backend", "")
        framework = state.get("framework", "")
        arch = state.get("arch", "")

        # 加载配置
        config = load_config(dsl, backend=backend)
        
        # 执行静态检查（SOL 格式暂不进行静态检查）
        bench_type = state.get("bench_type", "kernelbench")
        if bench_type == "sol":
            static_check_passed, static_check_error = True, ""
        else:
            static_check_passed, static_check_error = self._check_task_desc_static(task_desc)
        
        if not static_check_passed:
            # 静态检查失败，处理重试
            result, should_continue = self._handle_static_check_failure(
                static_check_error, task_desc, op_name, llm_reasoning,
                retry_count, max_retries, history_entry, prompt, state
            )
            return result, should_continue
        
        # 静态检查通过，执行运行时检查（SOL 格式暂不进行运行时检查）
        runtime_check_passed = True
        runtime_check_error = ""
        if bench_type != "sol":
            runtime_check_passed, runtime_check_error = await self._check_task_desc_runtime(
                task_desc=task_desc,
                op_name=op_name,
                framework=framework,
                backend=backend,
                arch=arch,
                dsl=dsl,
                config=config,
                timeout=60
            )
        
        # 检查运行时检查结果
        if runtime_check_passed:
            # 所有检查通过
            result = self._build_ready_state(
                task_desc, op_name, message, llm_reasoning, runtime_check_passed,
                history_entry, prompt, max_retries, state
            )
            return result, False
        else:
            # 运行时检查失败，处理重试
            result, should_continue = self._handle_runtime_check_failure(
                runtime_check_error, task_desc, op_name, llm_reasoning,
                retry_count, max_retries, history_entry, prompt, state
            )
            return result, should_continue
    
    async def run(self, state: OpTaskBuilderState) -> Dict[str, Any]:
        """执行需求理解和代码生成
        
        Args:
            state: OpTaskBuilderState状态，可能包含：
                - user_input: 用户需求
                - user_feedback: 用户反馈（可选，用于多轮交互或用户拒绝场景）
                - previous_state: 上一轮状态（可选，用于保留对话历史和生成的代码）
                - generated_task_desc: 之前生成的代码（可选，当用户拒绝时会保留）
                
        Returns:
            更新后的状态字典，包含以下字段：
                - status: 状态（READY/NEED_CLARIFICATION/NEED_MODIFICATION/UNSUPPORTED）
                - generated_task_desc: 生成的任务代码（KernelBench格式或SOL格式，status=READY时存在，调用方应展示给用户确认）
                - op_name: 算子名称（如：relu, matmul, layernorm等）
                - agent_message: 给用户的消息（所有状态都存在，调用方应展示给用户）
                - agent_reasoning: Agent的推理过程（用于调试和日志）
                - clarification_question: 澄清问题（status=NEED_CLARIFICATION时存在，与agent_message相同，便于调用方使用）
                - modification_suggestion: 修改建议（status=NEED_MODIFICATION时存在）
                - static_check_passed: 静态检查是否通过（status=READY或NEED_MODIFICATION时存在）
                - static_check_error: 静态检查错误信息（status=NEED_MODIFICATION时存在）
                - conversation_history: 对话历史（当前轮次的对话，会被LangGraph自动累积）
                - iteration: 当前轮次
                - op_task_builder_prompt: 使用的prompt（用于调试）
                
        状态说明（调用方使用指南）：
            - READY: 成功生成任务
                * 调用方应展示 generated_task_desc 给用户确认
                * 用户确认通过：将 generated_task_desc 作为新输入，重新送入调用方进行合法性校验与执行
                * 用户拒绝：提示用户补充说明，将用户反馈作为 user_feedback，previous_state 传入下一轮
                
            - NEED_CLARIFICATION: 信息不足，需要用户补充
                * 调用方应展示 agent_message（或 clarification_question）给用户，提示用户补充信息
                * agent_message 中应包含准确、完整的所有缺失信息列表，避免反复沟通
                * 用户提供新输入后，调用方应重新调用 OpTaskBuilder
                
            - NEED_MODIFICATION: 生成的代码有问题或用户拒绝，需要重新生成
                * 调用方应展示 agent_message 和 modification_suggestion 给用户
                * 通常由静态检查失败或用户拒绝READY任务触发
                * 用户提供反馈后，调用方应重新调用 OpTaskBuilder，传入 user_feedback 和 previous_state
                
            - UNSUPPORTED: 不支持的需求
                * 调用方应展示 agent_message 给用户，说明不支持的原因
                * 流程结束
        """
        # 兼容 dict 输入：确保 state 是 OpTaskBuilderState
        # 注意：TypedDict 不支持 isinstance 检查，所以直接对 dict 做转换
        if isinstance(state, dict):
            state = OpTaskBuilderState(**state)
        
        try:
            self.step_count += 1
            
            # 获取输入信息
            user_input = state.get("user_input", "")
            user_feedback = state.get("user_feedback", "")
            
            # 获取迭代和检查重试相关状态
            current_iteration = state.get("iteration", 0)
            max_iterations = state.get("max_iterations", 5)
            max_check_retries = state.get("max_check_retries", self.max_check_retries)
            
            # 检查是否达到最大迭代次数（整个 while True 循环算一次迭代）
            if current_iteration >= max_iterations:
                logger.warning(f"OpTaskBuilder: Reached max iterations ({max_iterations}), returning NEED_CLARIFICATION")
                return {
                    "status": OpTaskBuilderStatus.NEED_CLARIFICATION,
                    "agent_message": f"已达到最大交互次数（{max_iterations}次），请提供更清晰的需求描述。",
                    "agent_reasoning": f"Max iterations reached: {max_iterations}",
                    "iteration": current_iteration,
                    "check_retry_count": 0,
                    "max_check_retries": max_check_retries,
                }
            
            # 构建对话历史条目（当前轮次的用户输入）
            history_entry = {
                "role": "user",
                "content": user_input + (f"\n用户补充: {user_feedback}" if user_feedback else "")
            }
            
            # 检查重试循环（包括解析失败、静态检查和运行时检查）
            # 注意：整个 while True 循环在 max_iterations 层面只算一次迭代
            # 但每次循环在 max_check_retries 层面都会 +1
            while True:
                # 每次循环开始时，检查重试计数 +1
                check_retry_count = state.get("check_retry_count", 0) + 1
                
                # 如果是循环的第一轮（check_retry_count == 1），重置错误信息（为新的循环做准备）
                if check_retry_count == 1:
                    state["static_check_error"] = ""
                    state["runtime_check_error"] = ""
                
                # 检查是否达到最大检查重试次数
                if check_retry_count > max_check_retries:
                    logger.warning(f"OpTaskBuilder: Reached max check retries ({max_check_retries}) in this iteration, returning NEED_MODIFICATION")
                    # 获取累积的错误信息用于返回（用于 agent_message 和 modification_suggestion）
                    accumulated_static_error = state.get("static_check_error", "")
                    accumulated_runtime_error = state.get("runtime_check_error", "")
                    # 构建错误摘要
                    error_summary = ""
                    if accumulated_static_error:
                        error_summary += f"静态检查错误：{accumulated_static_error}"
                    if accumulated_runtime_error:
                        if error_summary:
                            error_summary += "\n"
                        error_summary += f"运行时检查错误：{accumulated_runtime_error}"
                    if not error_summary:
                        error_summary = "检查失败次数过多"
                    
                    return {
                        "status": OpTaskBuilderStatus.NEED_MODIFICATION,
                        "agent_message": f"代码生成过程中检查失败次数过多（已重试{max_check_retries}次）。请确认您的需求，我将重新生成。",
                        "modification_suggestion": error_summary,  # 使用累积的错误信息
                        "agent_reasoning": f"Max check retries reached: {max_check_retries}",
                        "iteration": current_iteration + 1,  # 整个循环结束，iteration +1
                        "check_retry_count": 0,  # 重置检查重试计数
                        "max_check_retries": max_check_retries,
                        "static_check_error": "",  # 重置错误信息，不影响下次用户交互
                        "runtime_check_error": "",  # 重置错误信息，不影响下次用户交互
                    }
                
                # 更新状态中的检查重试计数
                state["check_retry_count"] = check_retry_count
                # 准备 LLM 输入
                input_data = self._prepare_llm_input(state)
                
                # 调用 LLM 并解析输出
                parse_result = await self._call_llm_and_parse(input_data)
                
                # 处理解析错误
                if parse_result["parse_error"] is not None:
                    result, should_continue = self._handle_parse_error(
                        parse_result["parse_error"], check_retry_count, max_check_retries, history_entry, parse_result["prompt"], state
                    )
                    if not should_continue:
                        # 整个循环结束，iteration +1
                        result["iteration"] = current_iteration + 1
                        return result
                    # 继续循环（check_retry_count 已在循环开始时更新）
                    continue
                
                # 提取解析结果
                op_name = parse_result["op_name"]
                raw_status = parse_result["status"]
                task_desc = parse_result["task_desc"]
                message = parse_result["message"]
                llm_reasoning = parse_result["llm_reasoning"]
                prompt = parse_result["prompt"]
                
                # 规范化状态值
                # parser 要求 LLM 输出 "ready"，但 OpTaskBuilderStatus.READY = "success"
                _STATUS_NORMALIZE = {
                    "ready": OpTaskBuilderStatus.READY,
                    "success": OpTaskBuilderStatus.READY,
                    "need_clarification": OpTaskBuilderStatus.NEED_CLARIFICATION,
                    "need_modification": OpTaskBuilderStatus.NEED_MODIFICATION,
                    "unsupported": OpTaskBuilderStatus.UNSUPPORTED,
                }
                status = _STATUS_NORMALIZE.get(
                    (raw_status or "").lower().strip(),
                    raw_status
                )
                logger.debug(f"OpTaskBuilder: raw_status='{raw_status}' -> normalized='{status}'")
                
                # 根据状态处理
                if status == OpTaskBuilderStatus.READY and task_desc:
                    # 处理 READY 状态（静态检查 + 运行时检查）
                    result, should_continue = await self._handle_ready_status(
                        task_desc, op_name, message, llm_reasoning, prompt,
                        check_retry_count, max_check_retries, history_entry, state
                    )
                    if not should_continue:
                        # 整个循环结束，iteration +1
                        result["iteration"] = current_iteration + 1
                        return result
                    # 继续循环（check_retry_count 已在循环开始时更新）
                    continue
                
                elif status == OpTaskBuilderStatus.UNSUPPORTED:
                    # 不支持的需求，重置检查重试计数
                    logger.info(f"OpTaskBuilder: Unsupported request")
                    result = self._build_unsupported_state(
                        message, llm_reasoning, history_entry, prompt, max_check_retries
                    )
                    result["iteration"] = current_iteration + 1  # 整个循环结束，iteration +1
                    result["check_retry_count"] = 0  # 重置检查重试计数
                    result["static_check_error"] = ""  # 重置错误信息，不影响下次用户交互
                    result["runtime_check_error"] = ""  # 重置错误信息，不影响下次用户交互
                    return result
                
                else:
                    # 需要澄清，重置检查重试计数
                    logger.info(f"OpTaskBuilder: Need clarification")
                    result = self._build_clarification_state(
                        message, op_name, llm_reasoning, history_entry, prompt, max_check_retries
                    )
                    result["iteration"] = current_iteration + 1  # 整个循环结束，iteration +1
                    result["check_retry_count"] = 0  # 重置检查重试计数
                    result["static_check_error"] = ""  # 重置错误信息，不影响下次用户交互
                    result["runtime_check_error"] = ""  # 重置错误信息，不影响下次用户交互
                    return result
                
        except Exception as e:
            logger.error(f"OpTaskBuilder.run() failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                "status": OpTaskBuilderStatus.NEED_CLARIFICATION,
                "agent_message": f"处理请求时发生错误，请重试或提供更清晰的描述。错误: {str(e)}",
                "agent_reasoning": str(e),
                "iteration": state.get("iteration", 0) + 1,
                "check_retry_count": 0,  # 重置重试计数
                "runtime_check_passed": True,
                "runtime_check_error": "",
            }
            
