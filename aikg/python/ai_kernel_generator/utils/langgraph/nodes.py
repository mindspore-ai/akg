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

"""Node factory for wrapping Agents as LangGraph nodes."""

from ai_kernel_generator.utils.langgraph.state import KernelGenState
import logging
import asyncio
import json

logger = logging.getLogger(__name__)


class NodeFactory:
    """节点工厂：将 Agent 包装成 LangGraph 节点"""
    
    @staticmethod
    def create_designer_node(designer_instance, trace_instance, config: dict):
        """创建 Designer 节点函数"""
        async def designer_node(state: KernelGenState) -> dict:
            """Designer 节点：生成算法伪代码"""
            # 记录任务信息
            task_id = state.get('task_id', '0')
            op_name = state.get('op_name', 'unknown')
            logger.info(f"Task {task_id}, op_name: {op_name}, current_agent: designer")
            
            # 直接使用 state（KernelGenState 本质上是 dict）
            result, prompt, reasoning = await designer_instance.run(task_info=state)
            
            # 记录到 Trace
            trace_instance.insert_agent_record(
                agent_name="designer",
                result=result,
                prompt=prompt,
                reasoning=reasoning
            )
            
            # 解析结果（如果有 space_config）
            code = result
            space_config = None
            
            # 只在 hint 模式启用时才尝试提取 space_config_code
            enable_hint_mode = config.get("enable_hint_mode", False)
            
            # 使用与原 Task 相同的解析逻辑（ParserFactory.robust_parse）
            # 这样可以处理 LangChain 1.0 后的各种输出格式
            try:
                from ai_kernel_generator.utils.common_utils import ParserFactory
                
                # 获取 Designer 的 parser
                agent_parser = getattr(designer_instance, 'code_parser', None)
                if agent_parser:
                    # 使用 robust_parse 进行解析（与原 Task 保持一致）
                    parsed_result = ParserFactory.robust_parse(result, agent_parser)
                    if parsed_result:
                        # 从解析结果中提取 code
                        code = getattr(parsed_result, 'code', result)
                        # 只在 hint 模式下才提取 space_config_code
                        if enable_hint_mode:
                            space_config = getattr(parsed_result, 'space_config_code', None)
                else:
                    # 如果没有 parser，使用原始结果
                    logger.warning(f"[Task {state.get('task_id', '0')}] Designer 没有 parser，使用原始输出")
            except Exception as e:
                # 解析失败，使用原始结果
                logger.warning(f"[Task {state.get('task_id', '0')}] Designer JSON 解析失败: {e}，使用原始输出")
            
            # 计算新的 step_count
            new_step_count = state.get("step_count", 0) + 1
            
            updates = {
                "designer_code": code,
                "designer_prompt": prompt,
                "designer_reasoning": reasoning,
                "iteration": state.get("iteration", 0) + 1,
                "step_count": new_step_count,
                "agent_history": ["designer"]
            }
            
            # 优先从返回的 JSON 中提取，如果没有则从 state 中获取（兼容原来的直接修改方式）
            if space_config:
                updates["space_config_code"] = space_config
                logger.info(f"[Task {state.get('task_id', '0')}] Designer 从返回JSON中提取了 space_config_code (长度: {len(space_config)})")
            elif "space_config_code" in state and state["space_config_code"]:
                # Designer.run() 可能直接修改了 state（原来的行为）
                space_config = state["space_config_code"]
                updates["space_config_code"] = space_config
                logger.info(f"[Task {state.get('task_id', '0')}] Designer 从 state 中提取了 space_config_code (长度: {len(space_config)})")
            else:
                logger.info(f"[Task {state.get('task_id', '0')}] Designer 未生成 space_config_code")
            
            # 如果生成了 space_config，立即保存到文件（不依赖多 case 验证）
            if space_config:
                await NodeFactory._save_space_config(state, space_config, new_step_count, config)
            
            return updates
        
        return designer_node
    
    @staticmethod
    def create_coder_node(coder_instance, trace_instance):
        """创建 Coder 节点函数"""
        async def coder_node(state: KernelGenState) -> dict:
            """Coder 节点：生成可执行代码"""
            # 记录任务信息
            task_id = state.get('task_id', '0')
            op_name = state.get('op_name', 'unknown')
            logger.info(f"Task {task_id}, op_name: {op_name}, current_agent: coder")
            
            # 记录是否有错误信息传递给 Coder
            verifier_error = state.get('verifier_error', '')
            conductor_suggestion = state.get('conductor_suggestion', '')
            
            if verifier_error:
                task_id = state.get('task_id', '0')
                logger.info(f"[Task {task_id}] Coder 收到验证错误信息 (长度: {len(verifier_error)})")
                logger.debug(f"[Task {task_id}] 错误详情: {verifier_error[:200]}...")
            
            if conductor_suggestion:
                task_id = state.get('task_id', '0')
                logger.info(f"[Task {task_id}] Coder 收到 Conductor 建议 (长度: {len(conductor_suggestion)})")
                logger.debug(f"[Task {task_id}] 建议内容: {conductor_suggestion[:200]}...")
            
            # 直接使用 state（KernelGenState 本质上是 dict）
            result, prompt, reasoning = await coder_instance.run(task_info=state)
            
            # 使用与原 Task 相同的解析逻辑（ParserFactory.robust_parse）
            # 这样可以处理 LangChain 1.0 后的各种输出格式
            code = result
            try:
                from ai_kernel_generator.utils.common_utils import ParserFactory
                
                # 获取 Coder 的 parser
                agent_parser = getattr(coder_instance, 'code_parser', None)
                if agent_parser:
                    # 使用 robust_parse 进行解析（与原 Task 保持一致）
                    parsed_result = ParserFactory.robust_parse(result, agent_parser)
                    if parsed_result:
                        # 从解析结果中提取 code
                        code = getattr(parsed_result, 'code', result)
                        task_id = state.get('task_id', '0')
                        logger.info(f"[Task {task_id}] Coder 使用 robust_parse 解析成功 (原始长度: {len(result)}, 提取后长度: {len(code)})")
                    else:
                        task_id = state.get('task_id', '0')
                        logger.warning(f"[Task {task_id}] Coder robust_parse 返回空，使用原始输出")
                else:
                    # 如果没有 parser，使用原始结果
                    task_id = state.get('task_id', '0')
                    logger.warning(f"[Task {task_id}] Coder 没有 parser，使用原始输出")
            except Exception as e:
                # 解析失败，使用原始结果
                task_id = state.get('task_id', '0')
                logger.warning(f"[Task {task_id}] Coder JSON 解析失败: {e}，使用原始输出")
            
            trace_instance.insert_agent_record(
                agent_name="coder",
                result=result,  # 原始结果记录到 trace
                prompt=prompt,
                reasoning=reasoning
            )
            
            return {
                "coder_code": code,  # 解析后的纯代码
                "coder_prompt": prompt,
                "coder_reasoning": reasoning,
                "iteration": state.get("iteration", 0) + 1,
                "step_count": state.get("step_count", 0) + 1,
                "agent_history": ["coder"],
                "conductor_suggestion": None  # 清除旧建议
            }
        
        return coder_node
    
    @staticmethod
    def create_verifier_node(verifier_instance, device_pool, trace_instance, config,
                           private_worker=None, worker_manager=None, backend=None, arch=None):
        """创建 Verifier 节点函数（Worker 模式）"""
        # 捕获外层变量到闭包
        _private_worker = private_worker
        _worker_manager = worker_manager
        _backend = backend
        _arch = arch
        
        async def verifier_node(state: KernelGenState) -> dict:
            """Verifier 节点：验证代码正确性（包含单 case + 多 case + profile）"""
            # 记录任务信息
            task_id = state.get('task_id', '0')
            op_name = state.get('op_name', 'unknown')
            logger.info(f"Task {task_id}, op_name: {op_name}, current_agent: verifier")
            
            # 获取 Worker (兼容私有Worker和全局WorkerManager)
            worker = None
            if _private_worker:
                worker = _private_worker
            elif _worker_manager:
                worker = await _worker_manager.select(
                    backend=_backend,
                    arch=_arch
                )
            
            if not worker:
                raise RuntimeError(
                    f"No available worker for backend={_backend}, arch={_arch}. "
                    "Please register a worker first."
                )
            
            # 设置 verifier 的 worker
            verifier_instance.worker = worker
            
            try:
                current_step = state.get("step_count", 0)
                loop = asyncio.get_running_loop()
                
                # 单 case 验证（直接传递 state）
                # 注意：device_id 使用默认值 -1，由 Worker 内部管理
                verify_res, verify_log = await verifier_instance.run(
                    state,  # 直接使用 state
                    current_step,
                    device_id=-1  # Worker 模式默认使用 -1
                )
                
                # 多 case 验证（如果单 case 通过）
                multi_case_error = ""
                if verify_res and NodeFactory._should_run_multi_case(config):
                    task_id = state.get('task_id', '0')
                    logger.info(f"[Task {task_id}] Single case passed, starting multi-case verification...")
                    multi_res, multi_log = await NodeFactory._run_multi_case(
                        state, verifier_instance, current_step, config, trace_instance
                    )
                    if not multi_res:
                        task_id = state.get('task_id', '0')
                        logger.warning(f"[Task {task_id}] Multi-case verification failed")
                        verify_res = False
                        verify_log = f"单 case 通过，多 case 失败:\n{multi_log}"
                        multi_case_error = multi_log
                    else:
                        # 多 case 验证通过，清除错误
                        multi_case_error = ""
                
                # Profile（如果所有验证都通过）
                profile_res = {}
                task_type = state.get("task_type", "precision_only")
                backend = state.get("backend", "")
                if verify_res and task_type == "profile" and backend in ["ascend", "cuda"]:
                    task_id = state.get('task_id', '0')
                    logger.info(f"[Task {task_id}] All verifications passed, starting performance test...")
                    profile_res = await verifier_instance.run_profile(
                        state,  # 传递 state（兼容 task_info）
                        current_step,
                        -1,  # device_id，Worker 模式默认使用 -1
                        config.get("profile_settings", {})
                    )
                
                # 只有所有验证都通过后，才复制到 passed_cases
                if verify_res:
                    await loop.run_in_executor(
                        None,
                        NodeFactory._save_to_passed_cases,
                        state,
                        verifier_instance,
                        current_step,
                        config
                    )
                
                trace_instance.insert_agent_record(
                    agent_name="verifier",
                    result=str(verify_res),
                    error_log=verify_log,
                    profile_res=profile_res
                )
                
                # 记录验证结果
                if not verify_res:
                    task_id = state.get('task_id', '0')
                    logger.warning(f"[Task {task_id}] Verifier 失败，错误信息 (长度: {len(verify_log)})")
                    logger.debug(f"[Task {task_id}] 错误详情: {verify_log[:200]}...")
                
                return {
                    "verifier_result": verify_res,
                    "verifier_error": verify_log,
                    "profile_res": profile_res,  # 保留空字典，不转成 None
                    "multi_case_error": multi_case_error,  # 更新 multi_case_error
                    "step_count": state.get("step_count", 0) + 1,
                    "agent_history": ["verifier"]
                }
            finally:
                # 只有从 Manager 借来的才需要还
                if not _private_worker and _worker_manager:
                    await _worker_manager.release(worker)
        
        return verifier_node
    
    @staticmethod
    def create_conductor_node(trace_instance, config, conductor_template):
        """创建 Conductor 分析节点"""
        async def conductor_node(state: KernelGenState) -> dict:
            """Conductor 节点：分析错误并生成建议
            
            注意：此节点只在验证失败时被调用（由 verifier router 决定）
            """
            from ai_kernel_generator.core.llm.model_loader import create_model
            from ai_kernel_generator.utils.common_utils import ParserFactory
            from ai_kernel_generator.utils.result_processor import ResultProcessor
            
            # 记录任务信息
            task_id = state.get('task_id', '0')
            op_name = state.get('op_name', 'unknown')
            logger.info(f"Task {task_id}, op_name: {op_name}, current_agent: conductor")
            
            op_name = state.get("op_name", "")
            
            try:
                # 获取 Conductor 解析器
                conductor_parser = ParserFactory.get_conductor_parser()
                format_instructions = conductor_parser.get_format_instructions()
                
                # 准备历史记录
                history_for_analysis = []
                for attempt in state.get("history_attempts", [])[-5:]:  # 最近5次
                    history_for_analysis.append({
                        'code': attempt.get('code', '')[:2000],
                        'suggestion': attempt.get('suggestion', '')[:500]
                    })
                
                # 构建输入数据（与原 Conductor 一致）
                input_data = {
                    'dsl': state.get('dsl', ''),
                    'expert_suggestion': state.get('expert_suggestion', ''),
                    'op_name': op_name,
                    'framework': state.get('framework', ''),
                    'task_desc': state.get('task_desc', ''),
                    'agent_name': 'verifier',
                    'agent_result': state.get('coder_code', '')[:2000],
                    'error_log': state.get('verifier_error', '')[:5000],
                    'history_attempts': history_for_analysis,
                    'valid_next_agents': 'coder, finish',  # 固定选项
                    'format_instructions': format_instructions,
                }
                
                # 渲染模板
                prompt = conductor_template.render(**input_data)
                
                # 调用 LLM
                model_config = config.get("agent_model_config", {})
                model_name = model_config.get("conductor")
                if not model_name:
                    task_id = state.get('task_id', '0')
                    logger.warning(f"[Task {task_id}] No conductor model configured")
                    return {"conductor_suggestion": ""}
                
                model = create_model(model_name)
                
                # 判断模型类型并调用
                if hasattr(model, 'ainvoke'):
                    response = await model.ainvoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    reasoning = ""
                else:
                    completion = await model.chat.completions.create(
                        model=model.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=model.temperature,
                        max_tokens=model.max_tokens,
                        top_p=model.top_p,
                        **({"extra_body": model.extra_body} if hasattr(model, 'extra_body') and model.extra_body else {})
                    )
                    response_text = completion.choices[0].message.content
                    reasoning = ""
                
                # 解析结果
                agent_decision, suggestion = ResultProcessor.parse_conductor_decision(
                    response_text, conductor_parser, {"coder", "finish"}
                )
                
                # 保存到 trace（会自动保存到 txt 文件）
                trace_instance.insert_conductor_agent_record(
                    res=response_text,
                    prompt=prompt,
                    reasoning=reasoning,
                    agent_name="decision"
                )
                
                # 更新历史记录
                if state.get('coder_code') and state.get('verifier_error'):
                    history_entry = {
                        'code': state.get('coder_code', ''),
                        'error': state.get('verifier_error', ''),
                        'suggestion': suggestion if suggestion else '',
                        'task_desc': state.get('task_desc', '')
                    }
                    # history_attempts 会自动累积（因为定义为 Annotated[List, add]）
                    return {
                        "conductor_suggestion": suggestion or "",
                        "conductor_decision": agent_decision or "coder",
                        "history_attempts": [history_entry],
                        "agent_history": ["conductor"]
                    }
                
                task_id = state.get('task_id', '0')
                logger.info(f"[Task {task_id}] Conductor analysis completed, suggestion length: {len(suggestion or '')}")
                
                return {
                    "conductor_suggestion": suggestion or "",
                    "conductor_decision": agent_decision or "coder",
                    "agent_history": ["conductor"]
                }
                
            except Exception as e:
                task_id = state.get('task_id', '0')
                logger.error(f"[Task {task_id}] Conductor analysis failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return {
                    "conductor_suggestion": "",
                    "conductor_decision": "coder",
                    "agent_history": ["conductor"]
                }
        
        return conductor_node
    
    @staticmethod
    async def _save_space_config(state: KernelGenState, space_config_code: str, step_count: int, config: dict):
        """
        保存 space_config 到文件（独立于多 case 验证）
        
        Args:
            state: 当前状态
            space_config_code: 参数空间配置代码
            step_count: 当前步骤数（对应原来的 current_step）
            config: 配置字典
        """
        try:
            import os
            
            if not space_config_code:
                return
            
            # 确定保存目录
            log_dir = config.get('log_dir', '')
            if not log_dir:
                task_id = state.get('task_id', '0')
                logger.warning(f"[Task {task_id}] 未配置 log_dir，无法保存 space_config")
                return
            
            op_name = state.get('op_name')
            task_id = state.get('task_id', '0')
            
            expanded_log_dir = os.path.expanduser(log_dir)
            iter_dir = os.path.join(expanded_log_dir, op_name, f"I{task_id}_S{step_count:02d}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # 保存 space_config.py
            space_config_path = os.path.join(iter_dir, f"{op_name}_space_config.py")
            with open(space_config_path, 'w', encoding='utf-8') as f:
                f.write(space_config_code)
            
            task_id = state.get('task_id', '0')
            logger.info(f"[Task {task_id}] 保存参数空间配置: {space_config_path}")
            
        except Exception as e:
            task_id = state.get('task_id', '0')
            logger.warning(f"[Task {task_id}] 保存 space_config 失败: {e}")
    
    @staticmethod
    def _should_run_multi_case(config: dict) -> bool:
        """判断是否需要多 case 验证
        
        逻辑：
        1. Hint 模式：enable_hint_mode=True AND enable_multi_case_verification=True
        2. LLM 推理模式：enable_llm_range_inference=True (一定验证，不受 enable_multi_case_verification 控制)
        """
        enable_multi_case = config.get("enable_multi_case_verification", True)
        enable_hint = config.get("enable_hint_mode", False)
        enable_llm_inference = config.get("enable_llm_range_inference", False)
        
        logger.debug(f"多case验证配置检查: enable_multi_case_verification={enable_multi_case}, "
                    f"enable_hint_mode={enable_hint}, enable_llm_range_inference={enable_llm_inference}")
        
        # LLM 推理模式：一定执行多 case 验证
        if enable_llm_inference:
            return True
        
        # Hint 模式：需要同时开启 enable_hint_mode 和 enable_multi_case_verification
        if enable_hint and enable_multi_case:
            return True
        
        return False
    
    @staticmethod
    async def _run_multi_case(state, verifier_instance, step, config, trace):
        """运行多 case 验证（Hint 模式或 LLM 推理模式）
        
        Args:
            state: KernelGenState 状态
            verifier_instance: KernelVerifier 实例
            device_id: 设备 ID
            step: 当前步骤
            config: 配置字典
            trace: Trace 实例
            
        Returns:
            tuple[bool, str]: (验证结果, 错误日志)
        """
        enable_hint = config.get('enable_hint_mode', False)
        enable_llm_inference = config.get('enable_llm_range_inference', False)
        space_config_code = state.get("space_config_code")
        
        op_name = state.get('op_name')
        task_id = state.get('task_id', '0')
        logger.info(f"[Task {task_id}] 多case验证模式检查:")
        logger.info(f"[Task {task_id}]   enable_hint_mode: {enable_hint}")
        logger.info(f"[Task {task_id}]   enable_llm_range_inference: {enable_llm_inference}")
        logger.info(f"[Task {task_id}]   has_space_config_code: {bool(space_config_code)}")
        
        # 模式1：Hint模式（优先，需要 enable_hint=True 且有 space_config_code）
        if enable_hint and space_config_code:
            logger.info(f"[Task {task_id}] 使用Hint模式进行多case验证")
            return await NodeFactory._run_hint_mode_verification(
                state, verifier_instance, step, config
            )
        
        # 模式2：LLM推理模式（只需要 enable_llm_inference=True）
        if enable_llm_inference:
            logger.info(f"[Task {task_id}] 使用LLM推理模式进行多case验证")
            return await NodeFactory._run_llm_inference_mode_verification(
                state, verifier_instance, step, config, trace
            )
        
        # 都未启用或条件不满足
        if enable_hint and not space_config_code:
            logger.warning(f"[Task {task_id}] Hint模式已启用但未找到space_config_code，跳过多case验证")
        else:
            logger.info(f"[Task {task_id}] 未启用多case验证模式")
        
        return True, ""
    
    @staticmethod
    async def _run_hint_mode_verification(state, verifier_instance, step, config):
        """Hint模式的多case验证"""
        try:
            import os
            from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
            from ai_kernel_generator.utils.case_generator import MultiCaseGenerator
            
            op_name = state.get("op_name")
            task_id = state.get("task_id")
            framework = state.get("framework")
            dsl = state.get("dsl")
            backend = state.get("backend")
            arch = state.get("arch")
            
            # 1. 获取space_config_code
            space_config_code = state.get("space_config_code", "")
            if not space_config_code:
                task_id = state.get('task_id', '0')
                logger.warning(f"[Task {task_id}] 未找到space_config_code，跳过Hint模式验证")
                return True, ""
            
            # 2. 确定迭代目录
            log_dir = config.get('log_dir', '')
            if not log_dir:
                task_id = state.get('task_id', '0')
                logger.error(f"[Task {task_id}] 未配置log_dir，无法进行Hint模式验证")
                return False, "log_dir not configured"
            
            expanded_log_dir = os.path.expanduser(log_dir)
            iter_dir = os.path.join(expanded_log_dir, op_name, f"I{task_id}_multicase_S{step:02d}_verify")
            os.makedirs(iter_dir, exist_ok=True)
            
            # 3. 保存space_config.py
            space_config_path = os.path.join(iter_dir, f"{op_name}_space_config.py")
            with open(space_config_path, 'w', encoding='utf-8') as f:
                f.write(space_config_code)
            task_id = state.get('task_id', '0')
            logger.info(f"[Task {task_id}] 保存参数空间配置: {space_config_path}")
            
            # 4. 使用MultiCaseGenerator生成测试文件
            seed = config.get("sampling_seed", 42)
            generator = MultiCaseGenerator(space_config_path, seed=seed)
            multicase_file = os.path.join(iter_dir, f"{op_name}_multicase_{framework}.py")
            
            num_cases = config.get("multi_case_num", 10)
            strategy = config.get("sampling_strategy", "mixed")
            
            generator.generate_multicase_file(
                output_path=multicase_file,
                num_cases=num_cases,
                strategy=strategy
            )
            task_id = state.get('task_id', '0')
            logger.info(f"[Task {task_id}] 生成多case测试文件: {multicase_file}")
            
            # 5. 读取生成的多case测试文件内容
            with open(multicase_file, 'r', encoding='utf-8') as f:
                multicase_task_desc = f.read()
            
            # 6. 创建验证器并验证（直接使用 state）
            multi_case_verifier = KernelVerifier(
                op_name=op_name,
                framework_code=multicase_task_desc,  # 传递代码字符串
                task_id=f"{task_id}_multicase",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                impl_func_name=verifier_instance.impl_func_name,
                config=config,
                worker=verifier_instance.worker  # 新增：传递 worker
            )
            
            # 7. 运行验证（device_id 默认 -1，由 Worker 管理）
            multi_verify_res, multi_verify_log = await multi_case_verifier.run(
                state,  # 直接使用 state
                step,   # current_step
                device_id=-1  # Worker 模式默认使用 -1
            )
            
            # 8. 处理验证结果
            if multi_verify_res:
                task_id = state.get('task_id', '0')
                logger.info(f"[Task {task_id}] Hint模式多case验证通过")
                return True, ""
            else:
                task_id = state.get('task_id', '0')
                logger.warning(f"[Task {task_id}] Hint模式多case验证失败")
                return False, multi_verify_log
                
        except Exception as e:
            logger.error(f"[{state.get('op_name')}] Hint模式多case验证异常: {e}")
            import traceback
            error_log = traceback.format_exc()
            return False, error_log
    
    @staticmethod
    async def _run_llm_inference_mode_verification(state, verifier_instance, step, config, trace):
        """LLM推理模式的多case验证"""
        try:
            from ai_kernel_generator.core.agent.test_case_generator import TestCaseGenerator
            from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
            
            op_name = state.get("op_name")
            task_desc = state.get("task_desc")
            task_id = state.get("task_id")
            framework = state.get("framework")
            dsl = state.get("dsl")
            backend = state.get("backend")
            arch = state.get("arch")
            
            # 1. 创建 TestCaseGenerator
            test_gen = TestCaseGenerator(
                op_name=op_name,
                task_desc=task_desc,
                framework=framework,
                dsl=dsl,
                config=config
            )
            
            # 1.1 准备输入数据（包含之前的错误信息）
            # state 本身已包含所有字段，只需添加 previous_error
            state_with_error = dict(state)
            state_with_error["previous_error"] = state.get("multi_case_error", "")
            
            # 2. 生成多 case task_desc
            new_task_desc, prompt, reasoning = await test_gen.run(state_with_error)
            
            task_id = state.get('task_id', '0')
            logger.info(f"[Task {task_id}] 多 case task_desc 生成完成")
            
            # 2.1 记录 TestCaseGenerator 的执行结果
            trace.insert_agent_record(
                agent_name="test_case_generator",
                result=new_task_desc,
                prompt=prompt,
                reasoning=reasoning
            )
            
            # 3. 使用新 task_desc 创建临时 verifier 进行验证
            multi_case_verifier = KernelVerifier(
                op_name=op_name,
                framework_code=new_task_desc,  # 使用新的 task_desc
                task_id=f"{task_id}_multicase",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                impl_func_name=verifier_instance.impl_func_name,
                config=config,
                worker=verifier_instance.worker  # 新增：传递 worker
            )
            
            # 4. 运行验证（device_id 默认 -1，由 Worker 管理）
            multi_verify_res, multi_verify_log = await multi_case_verifier.run(
                state,  # 直接使用 state
                step,   # 使用单 case 验证的 step
                device_id=-1  # Worker 模式默认使用 -1
            )
            
            # 5. 更新 state 中的 multi_case_error
            if multi_verify_res:
                task_id = state.get('task_id', '0')
                logger.info(f"[Task {task_id}] 多 case 验证通过")
                # 验证通过，清除错误信息
                state["multi_case_error"] = ""
                return True, ""
            else:
                task_id = state.get('task_id', '0')
                logger.warning(f"[Task {task_id}] 多 case 验证失败")
                # 保存错误信息到 state，供下次迭代使用
                state["multi_case_error"] = multi_verify_log
                return False, multi_verify_log
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{state.get('op_name')}] 多 case 验证过程异常: {error_msg}")
            import traceback
            error_detail = traceback.format_exc()
            return False, f"多 case 验证异常: {error_msg}\n{error_detail}"
    
    @staticmethod
    def _save_to_passed_cases(state, verifier_instance, current_step: int, config: dict):
        """
        将验证通过的文件复制到 passed_cases 目录
        
        Args:
            state: 当前状态
            verifier_instance: Verifier 实例
            current_step: 当前步骤（用于构建目录名）
            config: 配置字典
        """
        try:
            import shutil
            import os
            from pathlib import Path
            
            log_dir = config.get('log_dir', '')
            if not log_dir:
                return
            
            op_name = state.get("op_name")
            
            # 构建源目录路径
            expanded_log_dir = os.path.expanduser(log_dir)
            unique_dir_name = f"I{verifier_instance.task_id}_S{current_step:02d}_verify"
            src_dir = os.path.join(expanded_log_dir, op_name, unique_dir_name)
            
            # 构建目标目录路径
            dst_dir = Path(log_dir).expanduser() / "passed_cases" / op_name / unique_dir_name
            
            # 复制目录
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                task_id = state.get('task_id', '0')
                logger.info(f"[Task {task_id}] 验证文件已保存到: {dst_dir}")
            else:
                task_id = state.get('task_id', '0')
                logger.warning(f"[Task {task_id}] 源目录不存在: {src_dir}")
        
        except Exception as e:
            logger.warning(f"[{state.get('op_name')}] 保存到 passed_cases 失败: {e}")