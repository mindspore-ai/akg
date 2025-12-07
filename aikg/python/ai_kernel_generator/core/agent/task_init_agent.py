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

"""TaskInitAgent: 将用户文字需求转换为KernelBench格式的Agent"""

import logging
import ast
from typing import Tuple, Dict, Any, Optional

from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory
from ai_kernel_generator.utils.langgraph.task_init_state import TaskInitState, TaskInitStatus

logger = logging.getLogger(__name__)


class TaskInitAgent(AgentBase):
    """将用户文字需求转换为KernelBench格式的Agent
    
    支持多轮交互：
    1. 理解用户的自然语言需求
    2. 生成标准的KernelBench格式代码
    3. 如果需求不清晰，提示用户补充信息
    4. 对生成的代码进行静态检查
    """
    
    def __init__(self, config: dict):
        """初始化TaskInitAgent
        
        Args:
            config: 配置字典，需包含agent_model_config等
        """
        context = {
            "agent_name": "task_init",
        }
        super().__init__(context=context, config=config)
        
        # 从config中获取model_config
        if config:
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for TaskInitAgent")
        
        # 加载prompt模板
        self.task_init_prompt = self.load_template("task_init/task_init.j2")
        
        # 获取解析器
        self.parser = ParserFactory.get_task_init_parser()
        self.format_instructions = self.parser.get_format_instructions()
        
        # 交互计数
        self.step_count = 0
    
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
    
    def _build_conversation_context(self, state: TaskInitState) -> str:
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
    
    async def run(self, state: TaskInitState) -> Dict[str, Any]:
        """执行需求理解和代码生成
        
        Args:
            state: TaskInitState状态
            
        Returns:
            更新后的状态字典
        """
        try:
            self.step_count += 1
            
            # 获取输入信息
            user_input = state.get("user_input", "")
            user_feedback = state.get("user_feedback", "")
            conversation_context = self._build_conversation_context(state)
            
            # 获取之前的静态检查错误（如果有）
            previous_error = state.get("static_check_error", "")
            previous_task_desc = state.get("generated_task_desc", "")
            
            # 构建prompt输入
            input_data = {
                "user_input": user_input,
                "user_feedback": user_feedback,
                "conversation_history": conversation_context,
                "previous_error": previous_error,
                "previous_task_desc": previous_task_desc,
                "format_instructions": self.format_instructions,
                "framework": state.get("framework", "torch"),
                "backend": state.get("backend", "cuda"),
            }
            
            # 更新context用于日志
            self.context.update({
                "hash": f"TaskInit@{self.step_count}",
                "step": self.step_count,
            })
            
            # 调用LLM
            model_name = self.model_config.get("task_init", "default")
            llm_result, prompt, reasoning = await self.run_llm(
                self.task_init_prompt, 
                input_data, 
                model_name
            )
            
            # 解析LLM输出
            try:
                parsed = ParserFactory.robust_parse(llm_result, self.parser)
                
                op_name = getattr(parsed, 'op_name', None)
                status = getattr(parsed, 'status', TaskInitStatus.NEED_CLARIFICATION)
                task_desc = getattr(parsed, 'task_desc', "")
                message = getattr(parsed, 'message', "")
                llm_reasoning = getattr(parsed, 'reasoning', reasoning)
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse LLM output: {parse_error}, using fallback")
                # 解析失败，返回需要澄清的状态
                return {
                    "status": TaskInitStatus.NEED_CLARIFICATION,
                    "agent_message": "抱歉，我没有完全理解您的需求。请您更详细地描述一下：\n1. 您想要实现什么算子？\n2. 输入数据的形状是什么？\n3. 使用什么框架（PyTorch/MindSpore）？",
                    "agent_reasoning": f"Parse error: {str(parse_error)}",
                    "task_init_prompt": prompt,
                    "iteration": state.get("iteration", 0) + 1,
                    "conversation_history": [{
                        "role": "user",
                        "content": user_input + (f"\n用户补充: {user_feedback}" if user_feedback else "")
                    }],
                }
            
            # 构建对话历史条目
            history_entry = {
                "role": "user",
                "content": user_input + (f"\n用户补充: {user_feedback}" if user_feedback else "")
            }
            
            # 根据状态处理
            if status == TaskInitStatus.READY and task_desc:
                # 执行静态检查
                check_passed, check_error = self._check_task_desc_static(task_desc)
                
                if check_passed:
                    # 静态检查通过
                    logger.info(f"TaskInit: Generated valid task_desc for op '{op_name}'")
                    return {
                        "status": TaskInitStatus.READY,
                        "generated_task_desc": task_desc,
                        "op_name": op_name,
                        "agent_message": message or f"已成功生成 {op_name} 算子的KernelBench格式代码。",
                        "agent_reasoning": llm_reasoning,
                        "static_check_passed": True,
                        "static_check_error": "",
                        "task_init_prompt": prompt,
                        "iteration": state.get("iteration", 0) + 1,
                        "conversation_history": [history_entry, {
                            "role": "assistant", 
                            "content": f"已生成 {op_name} 算子代码，格式验证通过。"
                        }],
                    }
                else:
                    # 静态检查失败，需要重新生成
                    logger.warning(f"TaskInit: Static check failed: {check_error}")
                    return {
                        "status": TaskInitStatus.NEED_MODIFICATION,
                        "generated_task_desc": task_desc,
                        "op_name": op_name,
                        "agent_message": f"生成的代码存在格式问题：{check_error}\n请确认您的需求，我将重新生成。",
                        "modification_suggestion": check_error,
                        "agent_reasoning": llm_reasoning,
                        "static_check_passed": False,
                        "static_check_error": check_error,
                        "task_init_prompt": prompt,
                        "iteration": state.get("iteration", 0) + 1,
                        "conversation_history": [history_entry, {
                            "role": "assistant",
                            "content": f"生成的代码格式检查未通过: {check_error}"
                        }],
                    }
            
            elif status == TaskInitStatus.UNSUPPORTED:
                # 不支持的需求
                logger.info(f"TaskInit: Unsupported request")
                return {
                    "status": TaskInitStatus.UNSUPPORTED,
                    "agent_message": message or "抱歉，当前系统不支持该类型的需求。AIKG主要用于生成高性能算子kernel代码。",
                    "agent_reasoning": llm_reasoning,
                    "task_init_prompt": prompt,
                    "iteration": state.get("iteration", 0) + 1,
                    "conversation_history": [history_entry, {
                        "role": "assistant",
                        "content": message or "该需求不在AIKG支持范围内。"
                    }],
                }
            
            else:
                # 需要澄清
                logger.info(f"TaskInit: Need clarification")
                return {
                    "status": TaskInitStatus.NEED_CLARIFICATION,
                    "clarification_question": message,
                    "agent_message": message or "请提供更多信息以便生成算子代码。",
                    "agent_reasoning": llm_reasoning,
                    "op_name": op_name,
                    "task_init_prompt": prompt,
                    "iteration": state.get("iteration", 0) + 1,
                    "conversation_history": [history_entry, {
                        "role": "assistant",
                        "content": message or "需要更多信息。"
                    }],
                }
                
        except Exception as e:
            logger.error(f"TaskInitAgent.run() failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                "status": TaskInitStatus.NEED_CLARIFICATION,
                "agent_message": f"处理请求时发生错误，请重试或提供更清晰的描述。错误: {str(e)}",
                "agent_reasoning": str(e),
                "iteration": state.get("iteration", 0) + 1,
            }
