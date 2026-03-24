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

"""算子专用路由工厂

实现算子生成场景的智能决策逻辑。
"""

from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.core_v2.langgraph_base.base_routers import (
    check_step_limit,
    check_agent_repeat_limit,
)
import logging

logger = logging.getLogger(__name__)


class RouterFactory:
    """算子路由工厂：实现算子场景的智能决策逻辑"""
    
    @staticmethod
    def create_verifier_router_with_conductor(config: dict):
        """Verifier 后的路由（决定是否需要 Conductor 分析）"""
        
        async def route_after_verifier(state: KernelGenState) -> str:
            # 1. 验证通过 → 直接结束，不需要 Conductor 分析
            if state.get("verifier_result"):
                logger.info("Verification passed, skipping conductor analysis, finishing task")
                return "finish"
            
            # 2. 验证失败 → 进入 Conductor 分析
            logger.info("Verification failed, routing to conductor for analysis")
            return "conductor"
        
        return route_after_verifier
    
    @staticmethod
    def create_conductor_router(config: dict, code_gen_agent: str = "coder",
                                enable_fix_code_gen: bool = False):
        """Conductor 分析后的路由决策（使用 Conductor 节点的决策）
        
        Args:
            config: 配置字典
            code_gen_agent: 代码生成 agent 名称（"coder" 或 "kernel_gen"）
            enable_fix_code_gen: 是否启用 fix_code_gen 增量修复路由
        """
        
        async def route_after_conductor(state: KernelGenState) -> str:
            # 1. 验证通过 → 结束
            if state.get("verifier_result"):
                logger.info("Verification passed, finishing task")
                return "finish"
            
            # 2. 检查步数限制
            max_step = config.get("max_step", 20)
            step_count = state.get("step_count", 0)
            if check_step_limit(step_count, max_step):
                logger.info(f"Reached max_step {max_step}, finishing task")
                return "finish"
            
            # 3. 检查限制
            agent_history = list(state.get("agent_history", []))
            repeat_limits = config.get("repeat_limits", {})
            
            possible_next = {code_gen_agent, "finish"}
            if enable_fix_code_gen:
                possible_next.add("fix_code_gen")

            illegal_agents = RouterFactory._check_agent_limits(
                step_count=step_count,
                max_step=max_step,
                agent_history=agent_history,
                repeat_limits=repeat_limits,
                code_gen_agent=code_gen_agent
            )
            
            # fix_code_gen 也受重复次数限制
            if enable_fix_code_gen and agent_history:
                max_fix_repeats = 3
                if repeat_limits and 'single_agent' in repeat_limits:
                    max_fix_repeats = repeat_limits['single_agent'].get("fix_code_gen", 3)
                if check_agent_repeat_limit(agent_history, "fix_code_gen", max_fix_repeats):
                    logger.info(f"fix_code_gen exceeds repeat limit {max_fix_repeats}")
                    illegal_agents.add("fix_code_gen")

            valid_next = possible_next - illegal_agents
            
            # 至少需要一个代码生成选项可用
            code_gen_options = {code_gen_agent}
            if enable_fix_code_gen:
                code_gen_options.add("fix_code_gen")
            if not (valid_next & code_gen_options):
                logger.info("No valid code gen agents available, finishing task")
                return "finish"
            
            # 4. 使用 Conductor 节点的决策结果
            conductor_decision = state.get("conductor_decision", code_gen_agent)
            if conductor_decision in valid_next:
                logger.info(f"Using conductor decision: {conductor_decision}")
                return conductor_decision
            
            # fallback: conductor 选了一个不可用的选项，用默认的代码生成 agent
            fallback = code_gen_agent if code_gen_agent in valid_next else "finish"
            logger.warning(
                f"Conductor decision '{conductor_decision}' not in valid options "
                f"{valid_next}, falling back to {fallback}"
            )
            return fallback
        
        return route_after_conductor
    
    @staticmethod
    def _check_agent_limits(step_count: int, max_step: int, 
                           agent_history: list, repeat_limits: dict,
                           code_gen_agent: str = "coder") -> set:
        """检查 agent 执行限制，返回被禁止的 agent 集合
        
        Args:
            step_count: 当前步数
            max_step: 最大步数
            agent_history: agent 执行历史
            repeat_limits: 重复次数限制配置
            code_gen_agent: 代码生成 agent 名称
            
        Returns:
            被禁止的 agent 集合
        """
        illegal_agents = set()
        
        # 检查总步数上限
        if check_step_limit(step_count, max_step):
            logger.info(f"Step count {step_count} exceeds max_step {max_step}, finishing task")
            return {code_gen_agent, "verifier", "conductor"}  # 全部禁止
        
        # 检查代码生成 agent 的连续重复次数（默认最多 3 次）
        if agent_history:
            max_repeats = 3
            if repeat_limits and 'single_agent' in repeat_limits:
                max_repeats = repeat_limits['single_agent'].get(code_gen_agent, 3)
            
            if check_agent_repeat_limit(agent_history, code_gen_agent, max_repeats):
                logger.info(f"{code_gen_agent} exceeds repeat limit {max_repeats}")
                illegal_agents.add(code_gen_agent)
        
        return illegal_agents
    
    @staticmethod
    async def _llm_decide_next(state, valid_options, template, model_config, trace):
        """使用 LLM 智能决策并分析错误（复用 Conductor 的 analyze.j2 逻辑）"""
        from akg_agents.core_v2.llm import create_llm_client
        from akg_agents.utils.common_utils import ParserFactory
        from akg_agents.op.utils.result_processor import ResultProcessor
        from akg_agents.cli.runtime.message_sender import send_message
        from akg_agents.cli.messages import DisplayMessage
        from akg_agents.utils.task_label import resolve_task_label
        import time

        session_id = state.get('session_id', '')
        task_id = str(state.get("task_id", ""))
        task_label = str(state.get("task_label") or "").strip()
        if not task_label:
            raise ValueError("[Router] state 中必须包含 task_label")
        start_time = time.time()
        if session_id:
            send_message(
                session_id,
                DisplayMessage(
                    text="[conductor] start",
                ),
            )
        
        try:
            # 获取 Conductor 解析器
            conductor_parser = ParserFactory.get_conductor_parser()
            format_instructions = conductor_parser.get_format_instructions()
            
            raw_error = state.get('verifier_error', '')
            error_for_prompt = raw_error
            if raw_error and len(raw_error) > 4000:
                error_for_prompt = "... (前面省略) ...\n" + raw_error[-4000:]

            input_data = {
                'dsl': state.get('dsl', ''),
                'expert_suggestion': state.get('expert_suggestion', ''),
                'op_name': state.get('op_name', ''),
                'framework': state.get('framework', ''),
                'task_desc': state.get('task_desc', ''),
                'agent_name': 'verifier',
                'agent_result': state.get('coder_code', ''),
                'error_log': error_for_prompt,
                'history_attempts': RouterFactory._format_history(state),
                'valid_next_agents': ', '.join(sorted(valid_options)),
                'format_instructions': format_instructions,
            }
            
            # 渲染模板
            prompt = template.render(**input_data)
            
            # 调用 LLM（使用 core_v2）
            model_level = model_config.get("conductor") or "standard"
            client = create_llm_client(model_level=model_level, session_id=session_id)
            
            # 调用 LLM
            messages = [{"role": "user", "content": prompt}]
            result = await client.generate(messages, stream=False)
            response_text = result.get("content", "")
            
            # 记录 LLM 调用到 trace（通用接口）
            trace.write_record("decision", [
                ('result', response_text),
                ('prompt', prompt),
            ], subdirectory="conductor")
            
            # 解析结果（使用 ResultProcessor，与 Conductor 一致）
            agent_decision, suggestion = ResultProcessor.parse_conductor_decision(
                response_text, conductor_parser, valid_options
            )
            
            if agent_decision:
                logger.info(f"LLM decided: {agent_decision}, suggestion: {suggestion[:100] if suggestion else 'None'}")
                if session_id:
                    duration = time.time() - start_time
                    summary = f"[conductor] done ({duration:.2f}s) decision={agent_decision}"
                    if suggestion:
                        summary += f" suggestion={suggestion}"
                    send_message(
                        session_id,
                        DisplayMessage(
                            text=summary,
                        ),
                    )

                return agent_decision, suggestion
            else:
                logger.warning(f"LLM decision not in valid options, using default")
            
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, using default")
            import traceback
            logger.debug(traceback.format_exc())
            if session_id:
                duration = time.time() - start_time
                send_message(
                    session_id,
                    DisplayMessage(
                        text=f"[conductor] error ({duration:.2f}s): {str(e)}",
                    ),
                )
        
        # 默认：如果有 coder 就返回 coder，否则 finish
        return ("coder" if "coder" in valid_options else "finish"), ""
    
    @staticmethod
    def _format_history(state):
        """格式化历史记录"""
        history = state.get("history_attempts", [])
        if not history:
            return []
        return [
            {
                'code': h.get('code', '')[:2000],
                'suggestion': h.get('suggestion', '')[:500]
            }
            for h in history[-5:]  # 最近 5 次
        ]
    
    @staticmethod
    def create_code_checker_router(config: dict, code_gen_agent: str = "coder",
                                    enable_fix_code_gen: bool = False):
        """创建 CodeChecker 后的路由决策

        CodeChecker 只做纯静态检查（ast.parse / py_compile / import），不涉及 LLM，
        因此不设最大重试次数——每次生成代码后都应该 check，失败就回去修。
        外层 workflow 的 max_iterations 已经能兜底防止死循环。

        当启用 fix_code_gen 时，语法错误优先走增量修复（更高效）；
        仅当 fix_code_gen 连续失败时回退到完整重新生成。

        Args:
            config: 配置字典
            code_gen_agent: 代码生成 agent 名称（默认 "coder"，KernelGen 流程使用 "kernel_gen"）
            enable_fix_code_gen: 是否启用 fix_code_gen 增量修复

        Returns:
            路由函数
        """
        async def route_after_code_checker(state: KernelGenState) -> str:
            task_id = state.get('task_id', '0')
            passed = state.get("code_check_passed", True)

            if passed:
                logger.info(f"[Task {task_id}] CodeChecker passed, routing to verifier")
                return "verifier"

            if enable_fix_code_gen:
                logger.info(
                    f"[Task {task_id}] CodeChecker failed, routing to fix_code_gen for incremental fix"
                )
                return "fix_code_gen"

            logger.info(
                f"[Task {task_id}] CodeChecker failed, routing back to {code_gen_agent} for fix"
            )
            return code_gen_agent

        return route_after_code_checker

    @staticmethod
    def create_codegen_router(next_agent: str, code_gen_agent: str = "coder"):
        """代码生成后的路由（处理 max_tokens 截断等异常）

        Args:
            next_agent: 正常情况下的下一节点（"verifier" 或 "code_checker"）
            code_gen_agent: 代码生成 agent 名称（"coder" 或 "kernel_gen"）
        """
        async def route_after_codegen(state: KernelGenState) -> str:
            task_id = state.get('task_id', '0')
            if state.get("codegen_invalid"):
                reason = state.get("codegen_invalid_reason", "")
                logger.warning(
                    f"[Task {task_id}] {code_gen_agent} 输出异常，路由至 conductor。"
                    f"{(' 原因: ' + reason) if reason else ''}"
                )
                return "conductor"
            logger.info(f"[Task {task_id}] {code_gen_agent} 输出正常，路由至 {next_agent}")
            return next_agent

        return route_after_codegen
    
    @staticmethod
    def create_smart_router(config, conductor_template, model_config):
        """智能路由器（用于 ConnectAll workflow）"""
        async def smart_route(state: KernelGenState) -> str:
            """根据当前状态智能决策下一步"""
            agent_history = state.get("agent_history", [])
            current_agent = agent_history[-1] if agent_history else None
            
            # 根据不同 agent 决定可选项
            if current_agent == "designer":
                options = {"coder", "finish"}
            elif current_agent == "coder":
                options = {"verifier", "designer", "finish"}
            elif current_agent == "verifier":
                if state.get("verifier_result"):
                    return "finish"
                options = {"coder", "designer", "finish"}
            else:
                return "finish"
            
            # 使用 LLM 决策
            return await RouterFactory._llm_decide_next(
                state, options, conductor_template, model_config
            )
        
        return smart_route
