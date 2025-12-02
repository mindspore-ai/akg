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

"""Router factory for intelligent decision-making in workflows."""

from ai_kernel_generator.utils.langgraph.state import KernelGenState
import logging

logger = logging.getLogger(__name__)


class RouterFactory:
    """路由工厂：实现智能决策逻辑"""
    
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
    def create_conductor_router(config: dict):
        """Conductor 分析后的路由决策（使用 Conductor 节点的决策）"""
        
        async def route_after_conductor(state: KernelGenState) -> str:
            # 1. 验证通过 → 结束
            if state.get("verifier_result"):
                logger.info("Verification passed, finishing task")
                return "finish"
            
            # 2. 检查步数限制
            max_step = config.get("max_step", 20)
            step_count = state.get("step_count", 0)
            if step_count >= max_step:
                logger.info(f"Reached max_step {max_step}, finishing task")
                return "finish"
            
            # 3. 检查限制
            agent_history = list(state.get("agent_history", []))
            repeat_limits = config.get("repeat_limits", {})
            
            possible_next = {"coder", "finish"}
            illegal_agents = RouterFactory._check_agent_limits(
                step_count=step_count,
                max_step=max_step,
                agent_history=agent_history,
                repeat_limits=repeat_limits
            )
            
            valid_next = possible_next - illegal_agents
            
            if not valid_next or "coder" not in valid_next:
                logger.info("No valid next agents, finishing task")
                return "finish"
            
            # 4. 使用 Conductor 节点的决策结果
            conductor_decision = state.get("conductor_decision", "coder")
            if conductor_decision in valid_next:
                logger.info(f"Using conductor decision: {conductor_decision}")
                return conductor_decision
            
            # 默认策略
            logger.warning(f"Conductor decision '{conductor_decision}' not in valid options, using coder")
            return "coder"
        
        return route_after_conductor
    
    @staticmethod
    def _check_agent_limits(step_count: int, max_step: int, 
                           agent_history: list, repeat_limits: dict) -> set:
        """检查 agent 执行限制，返回被禁止的 agent 集合
        
        Args:
            step_count: 当前步数
            max_step: 最大步数
            agent_history: agent 执行历史
            repeat_limits: 重复次数限制配置
            
        Returns:
            被禁止的 agent 集合
        """
        illegal_agents = set()
        
        # 检查总步数上限
        if step_count >= max_step:
            logger.info(f"Step count {step_count} exceeds max_step {max_step}, finishing task")
            return {"coder", "verifier", "conductor"}  # 全部禁止
        
        # 检查 coder 的连续重复次数（默认最多 3 次）
        if agent_history:
            max_repeats = 3
            if repeat_limits and 'single_agent' in repeat_limits:
                max_repeats = repeat_limits['single_agent'].get('coder', 3)
            
            # 统计 coder 连续出现的次数
            consecutive_count = 0
            for agent in reversed(agent_history):
                if agent == "coder":
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= max_repeats:
                logger.info(f"Coder consecutive count {consecutive_count} exceeds limit {max_repeats}")
                illegal_agents.add("coder")
        
        return illegal_agents
    
    @staticmethod
    async def _llm_decide_next(state, valid_options, template, model_config, trace):
        """使用 LLM 智能决策并分析错误（复用 Conductor 的 analyze.j2 逻辑）"""
        from ai_kernel_generator.core.llm.model_loader import create_model
        from ai_kernel_generator.utils.common_utils import ParserFactory
        from ai_kernel_generator.utils.result_processor import ResultProcessor
        
        try:
            # 获取 Conductor 解析器
            conductor_parser = ParserFactory.get_conductor_parser()
            format_instructions = conductor_parser.get_format_instructions()
            
            # 构建输入数据（类似 Conductor._llm_decide_next_agent）
            input_data = {
                'dsl': state.get('dsl', ''),
                'expert_suggestion': state.get('expert_suggestion', ''),  # 添加 expert_suggestion
                'op_name': state.get('op_name', ''),
                'framework': state.get('framework', ''),
                'task_desc': state.get('task_desc', ''),
                'agent_name': 'verifier',
                'agent_result': state.get('coder_code', '')[:2000],
                'error_log': state.get('verifier_error', '')[:5000],
                'history_attempts': RouterFactory._format_history(state),
                'valid_next_agents': ', '.join(sorted(valid_options)),
                'format_instructions': format_instructions,
            }
            
            # 渲染模板
            prompt = template.render(**input_data)
            
            # 调用 LLM
            model_name = model_config.get("conductor")
            if not model_name:
                logger.warning("No conductor model configured, using default routing")
                return ("coder" if "coder" in valid_options else "finish"), ""
                
            model = create_model(model_name)
            
            # 判断模型类型并调用相应的方法
            if hasattr(model, 'ainvoke'):
                # Langchain 模型 (ChatDeepSeek, ChatOllama)
                response = await model.ainvoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                # AsyncOpenAI 模型 (vllm, 环境变量模式)
                completion = await model.chat.completions.create(
                    model=model.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=model.temperature,
                    max_tokens=model.max_tokens,
                    top_p=model.top_p,
                    **({"extra_body": model.extra_body} if hasattr(model, 'extra_body') and model.extra_body else {})
                )
                response_text = completion.choices[0].message.content
            
            # 记录 LLM 调用到 trace
            trace.insert_conductor_agent_record(
                res=response_text,
                prompt=prompt,
                reasoning="",  # AsyncOpenAI 没有 reasoning
                agent_name="decision"
            )
            
            # 解析结果（使用 ResultProcessor，与 Conductor 一致）
            agent_decision, suggestion = ResultProcessor.parse_conductor_decision(
                response_text, conductor_parser, valid_options
            )
            
            if agent_decision:
                logger.info(f"LLM decided: {agent_decision}, suggestion: {suggestion[:100] if suggestion else 'None'}")
                return agent_decision, suggestion
            else:
                logger.warning(f"LLM decision not in valid options, using default")
            
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, using default")
            import traceback
            logger.debug(traceback.format_exc())
        
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

