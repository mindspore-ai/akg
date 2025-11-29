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

import os
import json
import logging
import asyncio
from typing import Tuple, Optional, Dict, Any, List
from ai_kernel_generator import get_project_root
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.utils import check_task_config, check_task_type
from ai_kernel_generator.core.agent.conductor import Conductor
from ai_kernel_generator.core.agent.coder import Coder
from ai_kernel_generator.core.agent.designer import Designer
from ai_kernel_generator.core.agent.test_case_generator import TestCaseGenerator
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.utils.workflow_manager import WorkflowManager
from ai_kernel_generator.utils.collector import get_collector
from ai_kernel_generator.core.worker.manager import get_worker_manager

logger = logging.getLogger(__name__)


class Task:
    """
    配置驱动的任务类，基于workflow.yaml进行工作流管理
    """

    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 task_id: str,
                 backend: str,
                 arch: str,
                 dsl: str,
                 config: dict,
                 device_pool: Optional[DevicePool] = None,
                 framework: str = "torch",
                 task_type="precision_only",
                 workflow: Optional[str] = None,
                 inspirations: Optional[List[str]] = None,
                 meta_prompts: Optional[str] = None,
                 handwrite_suggestions: Optional[List[Dict[str, str]]] = None) -> None:
        """
        初始化Task类，基于workflow配置进行工作流管理。

        Args:
            op_name (str): 算子名称。
            task_desc (str): 算子描述。
            task_id (str): 任务ID。
            backend (str): 后端名称。
            arch (str): 架构名称。
            dsl (str): 实现类型。
            config (dict): 配置agent_mode_config。
            device_pool: 设备池（可选，用于向后兼容）。
            framework (str): 框架名称。
            task_type (str, optional): 任务类型, 默认为"precision_only"。
            workflow (str, optional): workflow名称，可以是文件名或完整路径。
                - 文件名: "coder_only_workflow" -> "config/coder_only_workflow.yaml"
                - 完整路径: "config/xxx.yaml"
                - None: 使用默认配置
            inspirations (List[str], optional): 启发示例列表。
            meta_prompts (str, optional): 元提示。
            handwrite_suggestions (List[Dict[str, str]], optional): 手写优化建议列表。
        """
        # 验证任务配置并规范化DSL（自动转换triton为triton_cuda或triton_ascend）
        normalized_dsl = check_task_config(framework, backend, arch, dsl)
        check_task_type(task_type)

        # 基础属性
        self.op_name = op_name
        self.task_desc = task_desc
        self.task_id = task_id
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.dsl = normalized_dsl.lower()  # 使用规范化后的DSL
        self.framework = framework.lower()
        self.task_type = task_type
        self.device_pool = device_pool
        self.inspirations = inspirations
        self.meta_prompts = meta_prompts
        self.handwrite_suggestions = handwrite_suggestions if handwrite_suggestions else []

        # 统一保存config，后续向下传递
        self.config = config

        # 兼容旧代码：如果提供了device_pool，创建私有Worker
        self._private_worker = None
        if self.device_pool:
            import warnings
            warnings.warn(
                "⚠️  [DEPRECATED] 直接传递 device_pool 给 Task() 是旧写法，将在未来版本移除。\n"
                "推荐的新写法：\n"
                "  1. 注册 LocalWorker 到 WorkerManager（一行代码）：\n"
                "     from ai_kernel_generator.core.worker.manager import register_local_worker\n"
                "     \n"
                "     await register_local_worker([0], backend='cuda', arch='a100')\n"
                "  2. 创建 Task 时不传 device_pool：\n"
                "     task = Task(\n"
                "         ...,\n"
                "         # device_pool=device_pool,  # 不再传递\n"
                "         ...\n"
                "     )\n"
                "参考示例：examples/run_torch_npu_triton_single.py",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning("⚠️  检测到使用旧的 device_pool 参数，请参考日志中的警告信息迁移到新写法")
            
            from ai_kernel_generator.core.worker.local_worker import LocalWorker
            self._private_worker = LocalWorker(self.device_pool, backend=self.backend)

        # 优先使用传入的workflow参数，否则使用config中的workflow_config_path
        if workflow:
            self.workflow_config_path = WorkflowManager.resolve_workflow_config_path(workflow)
        else:
            self.workflow_config_path = config.get("workflow_config_path")
            if self.workflow_config_path and not os.path.isabs(self.workflow_config_path):
                # 相对路径需要相对于项目根目录
                self.workflow_config_path = os.path.join(get_project_root(), self.workflow_config_path)

        # 确保workflow_config_path不为空
        if not self.workflow_config_path:
            raise ValueError("workflow_config_path is required. Please provide it in config or as workflow parameter.")

        # 初始化Conductor（优先初始化，用于获取workflow配置）
        self.conductor = Conductor(self.op_name, self.task_desc, self.task_id,
                                   self.dsl, self.framework, self.arch,
                                   workflow_config_path=self.workflow_config_path, config=self.config)

        # 根据workflow配置动态初始化agents
        self._init_agents_from_workflow()

    def _init_agents_from_workflow(self):
        """根据workflow.yaml配置动态初始化需要的agents"""
        # 获取workflow中配置的agent列表
        agent_names = set(self.conductor.agent_info.keys())

        # 初始化所需的agents
        self.agents = {}  # 存储所有agents的字典

        # 根据配置创建相应的agents
        if 'designer' in agent_names:
            self.designer = Designer(self.op_name, self.task_desc,
                                     self.dsl, self.backend, self.arch,
                                     workflow_config_path=self.workflow_config_path, config=self.config)
            self.agents['designer'] = self.designer

        if 'coder' in agent_names:
            self.coder = Coder(self.op_name, self.task_desc,
                               self.dsl, self.framework, self.backend, self.arch,
                               workflow_config_path=self.workflow_config_path, config=self.config)
            self.agents['coder'] = self.coder

        if 'verifier' in agent_names:
            self.verifier = KernelVerifier(self.op_name, self.task_desc,
                                           self.task_id, self.framework, self.dsl, self.backend, self.arch, config=self.config)
            self.agents['verifier'] = self.verifier

    def get_agent(self, agent_name: str):
        """获取指定名称的agent实例"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' is not available in current workflow configuration. "
                             f"Available agents: {list(self.agents.keys())}")
        return self.agents[agent_name]

    def init_conductor(self, init_task_info: Optional[Dict[str, Any]] = None):
        """
        初始化Conductor，根据初始任务信息进行初始化。

        Args:
            init_task_info (Dict[str, Any], optional): 初始任务信息字典, 包含初始代码等
        """

        # 初始化基础文档
        base_doc = {"backend": self.backend, "arch": self.arch, "dsl": self.dsl, "framework": self.framework}

        # 添加workflow名称
        workflow_name = os.path.basename(self.workflow_config_path).replace(
            '.yaml', '') if self.workflow_config_path else ""
        base_doc["workflow_name"] = workflow_name

        # 只在相应agent存在时添加其基础文档
        if 'designer' in self.agents:
            base_doc.update(self.agents['designer'].base_doc)

        if 'coder' in self.agents:
            coder = self.agents['coder']
            base_doc.update(coder.base_doc)

        # 初始化任务信息
        self.conductor.set_task_info(base_doc)

        # inspirations and meta_prompts from evolution
        self.conductor.task_info.update({"inspirations": self.inspirations})
        self.conductor.task_info.update({"meta_prompts": self.meta_prompts})
        self.conductor.task_info.update({"handwrite_suggestions": self.handwrite_suggestions})

        # 插入初始记录（如果有初始代码）
        # 注意：这里的逻辑假设从某个中间步骤开始，需要预先插入之前步骤的结果
        if init_task_info and init_task_info.get("designer_code"):
            self.conductor.record_agent_execution(
                agent_name="designer",
                result=json.dumps({"code": init_task_info.get("designer_code")})
            )

        if init_task_info and init_task_info.get("coder_code"):
            self.conductor.record_agent_execution(
                agent_name="coder",
                result=json.dumps({"code": init_task_info.get("coder_code")})
            )

    async def run(self, init_task_info: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, dict]:
        """
        异步运行任务，执行操作任务字符串

        Args:
            init_task_info: 初始任务信息字典，包含初始代码等

        Returns:
            Tuple[str, bool, dict]: (算子名称, 是否成功, 任务信息)
        """
        try:
            # 初始化conductor
            self.init_conductor(init_task_info)

            # 获取首个agent（通过yaml配置）
            current_agent = self.conductor.start_agent

            while current_agent != "finish":
                logger.info(f"Task {self.task_id}, op_name: {self.op_name}, current_agent: {current_agent}")
                try:
                    if current_agent == "designer":
                        designer = self.get_agent('designer')

                        designer_res, designer_prompt, designer_reasoning = await designer.run(
                            task_info=self.conductor.task_info
                        )
                        self.conductor.record_agent_execution(
                            agent_name="designer",
                            result=designer_res,
                            prompt=designer_prompt,
                            reasoning=designer_reasoning
                        )
                        
                        # 如果生成了space_config，立即保存到文件（不依赖多case验证）
                        if "space_config_code" in self.conductor.task_info:
                            # 获取当前步骤数
                            step = len(self.conductor.trace.trace_list)
                            await self._save_space_config(step)

                    elif current_agent == "coder":
                        coder = self.get_agent('coder')

                        coder_res, coder_prompt, coder_reasoning = await coder.run(
                            task_info=self.conductor.task_info
                        )

                        self.conductor.record_agent_execution(
                            agent_name="coder",
                            result=coder_res,
                            prompt=coder_prompt,
                            reasoning=coder_reasoning
                        )

                    elif current_agent == "verifier":
                        # 获取 Worker (兼容私有Worker和全局WorkerManager)
                        worker = None
                        if getattr(self, '_private_worker', None):
                            worker = self._private_worker
                        else:
                            worker = await get_worker_manager().select(
                                backend=self.backend,
                                arch=self.arch
                            )
                            
                        if not worker:
                            raise RuntimeError(f"No available worker for backend={self.backend}, arch={self.arch}. Please register a worker first.")
                        
                        self.verifier.worker = worker
                        
                        try:
                            current_step = len(self.conductor.trace.trace_list)
                            
                            # 直接异步调用 verifier.run
                            # device_id 使用默认值 -1，verifier 内部会自动管理：
                            # - LocalWorker: 从 device_pool 获取设备
                            # - RemoteWorker: 使用 0 作为占位符（远程服务器管理设备）
                            verify_res, verify_log = await self.verifier.run(
                                self.conductor.task_info, current_step
                            )
                            
                            profile_res = {}
                            if verify_res and self.task_type == "profile" and self.backend in ["ascend", "cuda"]:
                                logger.info(f"[{self.op_name}] 所有验证通过，开始性能测试（使用原始输入）...")
                                profile_settings = self.config.get("profile_settings", {})
                                profile_res = await self.verifier.run_profile(
                                    self.conductor.task_info, current_step, profile_settings=profile_settings
                                )
                            
                            # 只有所有验证都通过后，才复制到 passed_cases
                            if verify_res:
                                self._save_to_passed_cases_sync(current_step)
                            
                            self.conductor.record_agent_execution(
                                agent_name="verifier",
                                result=str(verify_res),
                                error_log=verify_log,
                                profile_res=profile_res
                            )
                        finally:
                            # 只有从 Manager 借来的才需要还
                            if not getattr(self, '_private_worker', None) and worker:
                                await get_worker_manager().release(worker)
                    else:
                        raise ValueError(f"Unsupported agent: {current_agent}")

                    # 获取下一个agent
                    next_agent = await self.conductor.get_next_agent()
                    current_agent = next_agent

                except Exception as agent_error:
                    logger.error(f"Agent {current_agent} execution failed: {agent_error}")
                    logger.error(f"Error type: {type(agent_error).__name__}")
                    logger.error(f"Full error details: {repr(agent_error)}")

                    self.conductor.record_agent_execution(
                        agent_name=current_agent,
                        result=f"ERROR: Agent execution failed: {str(agent_error)}",
                        error_log=str(agent_error)
                    )
                    return self.op_name, False, self.conductor.task_info

            # 获取最终结果
            final_success = self.conductor.task_info.get('verifier_result', False)

            if os.getenv("AIKG_DATA_COLLECT", "off").lower() == "on":
                try:
                    collector = await get_collector()
                    collector.set_config(self.config)

                    # 根据验证结果准备数据
                    if final_success:
                        # 验证成功：收集task相关数据 + database数据
                        saved_files = await collector.prepare_and_remove_data(task_id=self.task_id)
                        database_file = collector.prepare_database_data(self.conductor.task_info)
                        all_files = saved_files + ([database_file] if database_file else [])
                        logger.debug(
                            f"Task {self.task_id} completed successfully, saved {len(all_files)} files: {all_files}")
                    else:
                        # 验证失败：收集无task_id的独立数据
                        saved_files = await collector.prepare_and_remove_data()
                        logger.debug(f"Task {self.task_id} failed, saved {len(saved_files)} files: {saved_files}")
                except Exception as e:
                    logger.error(f"Failed to prepare data for transmission in task {self.task_id}: {e}")

            return self.op_name, final_success, self.conductor.task_info

        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return self.op_name, False, self.conductor.task_info
    
    def _save_to_passed_cases_sync(self, current_step: int):
        """
        将验证通过的文件复制到 passed_cases 目录
        
        Args:
            current_step: 当前步骤（用于构建目录名）
        """
        try:
            import shutil
            from pathlib import Path
            
            log_dir = self.config.get('log_dir', '')
            if not log_dir:
                return
            
            # 构建源目录路径
            expanded_log_dir = os.path.expanduser(log_dir)
            unique_dir_name = f"I{self.verifier.task_id}_S{current_step:02d}_verify"
            src_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir_name)
            
            # 构建目标目录路径
            dst_dir = Path(log_dir) / "passed_cases" / self.op_name / unique_dir_name
            
            # 复制目录
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                logger.info(f"[{self.op_name}] 验证文件已保存到: {dst_dir}")
            else:
                logger.warning(f"[{self.op_name}] 源目录不存在: {src_dir}")
        
        except Exception as e:
            logger.warning(f"[{self.op_name}] 保存到 passed_cases 失败: {e}")
    
    async def _save_space_config(self, current_step: int):
        """
        保存space_config到文件（独立于多case验证）
        
        Args:
            current_step: 当前迭代步骤
        """
        try:
            import os
            
            space_config_code = self.conductor.task_info.get("space_config_code", "")
            if not space_config_code:
                return
            
            # 确定保存目录
            log_dir = self.config.get('log_dir', '')
            if not log_dir:
                logger.warning(f"[{self.op_name}] 未配置log_dir，无法保存space_config")
                return
            
            expanded_log_dir = os.path.expanduser(log_dir)
            # 保存到迭代目录下
            iter_dir = os.path.join(expanded_log_dir, self.op_name, f"I{self.task_id}_S{current_step:02d}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # 保存space_config.py
            space_config_path = os.path.join(iter_dir, f"{self.op_name}_space_config.py")
            with open(space_config_path, 'w', encoding='utf-8') as f:
                f.write(space_config_code)
            
            logger.info(f"[{self.op_name}] 保存参数空间配置: {space_config_path}")
            
        except Exception as e:
            logger.warning(f"[{self.op_name}] 保存space_config失败: {e}")
    
    def _should_run_multi_case_test(self) -> bool:
        """
        判断是否需要运行多 case 测试
        
        Returns:
            bool: 是否需要运行多 case 测试
        """
        # 首先检查多 case 验证的总开关
        enable_multi_case_verification = self.config.get('enable_multi_case_verification', True)
        
        if not enable_multi_case_verification:
            logger.info(f"[{self.op_name}] 多 case 验证已禁用（enable_multi_case_verification=false）")
            return False
        
        # 如果总开关开启，检查具体模式
        enable_hint = self.config.get('enable_hint_mode', False)
        enable_llm_inference = self.config.get('enable_llm_range_inference', False)
        
        if enable_hint or enable_llm_inference:
            if enable_hint:
                logger.info(f"[{self.op_name}] Hint模式已启用，将在单 case 验证通过后运行")
            if enable_llm_inference:
                logger.info(f"[{self.op_name}] LLM推理模式已启用，将在单 case 验证通过后运行")
            return True
        else:
            logger.debug(f"[{self.op_name}] 多 case 测试模式未启用")
            return False
    
    async def _run_multi_case_verification(self, device_id: int, verify_step: int) -> tuple[bool, str]:
        """
        运行多 case 验证 - 支持两种模式
        
        Args:
            device_id: 设备 ID
            verify_step: 验证步骤
            
        Returns:
            tuple[bool, str]: (验证结果, 错误日志)
        """
        enable_hint = self.config.get('enable_hint_mode', False)
        enable_llm_inference = self.config.get('enable_llm_range_inference', False)
        
        # ============ 模式1：Hint模式（优先） ============
        if enable_hint and "space_config_code" in self.conductor.task_info:
            logger.info(f"[{self.op_name}] 使用Hint模式进行多case验证")
            return await self._run_hint_mode_verification(device_id, verify_step)
        
        # ============ 模式2：LLM推理模式 ============
        if enable_llm_inference:
            logger.info(f"[{self.op_name}] 使用LLM推理模式进行多case验证")
            return await self._run_llm_inference_mode_verification(device_id, verify_step)
        
        # 都未启用
        return True, ""
    
    async def _run_hint_mode_verification(self, device_id: int, verify_step: int) -> tuple[bool, str]:
        """Hint模式的多case验证"""
        try:
            import os
            from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
            from ai_kernel_generator.utils.case_generator import MultiCaseGenerator
            
            # 1. 获取space_config_code
            space_config_code = self.conductor.task_info.get("space_config_code", "")
            if not space_config_code:
                logger.warning(f"[{self.op_name}] 未找到space_config_code，跳过Hint模式验证")
                return True, ""
            
            # 2. 确定迭代目录
            log_dir = self.config.get('log_dir', '')
            if not log_dir:
                logger.error(f"[{self.op_name}] 未配置log_dir，无法进行Hint模式验证")
                return False, "log_dir not configured"
            
            expanded_log_dir = os.path.expanduser(log_dir)
            iter_dir = os.path.join(expanded_log_dir, self.op_name, f"I{self.task_id}_multicase_S{verify_step:02d}_verify")
            os.makedirs(iter_dir, exist_ok=True)
            
            # 3. 保存space_config.py
            space_config_path = os.path.join(iter_dir, f"{self.op_name}_space_config.py")
            with open(space_config_path, 'w', encoding='utf-8') as f:
                f.write(space_config_code)
            logger.info(f"[{self.op_name}] 保存参数空间配置: {space_config_path}")
            
            # 4. 使用MultiCaseGenerator生成测试文件
            seed = self.config.get("sampling_seed", 42)
            generator = MultiCaseGenerator(space_config_path, seed=seed)
            multicase_file = os.path.join(iter_dir, f"{self.op_name}_multicase_{self.framework}.py")
            
            num_cases = self.config.get("multi_case_num", 10)
            strategy = self.config.get("sampling_strategy", "mixed")
            
            generator.generate_multicase_file(
                output_path=multicase_file,
                num_cases=num_cases,
                strategy=strategy
            )
            logger.info(f"[{self.op_name}] 生成多case测试文件: {multicase_file}")
            
            # 5. 读取生成的多case测试文件内容
            with open(multicase_file, 'r', encoding='utf-8') as f:
                multicase_task_desc = f.read()
            
            # 6. 创建验证器并验证
            multi_case_verifier = KernelVerifier(
                op_name=self.op_name,
                framework_code=multicase_task_desc,  # 传递代码字符串
                task_id=f"{self.task_id}_multicase",
                framework=self.framework,
                dsl=self.dsl,
                backend=self.backend,
                arch=self.arch,
                impl_func_name=self.verifier.impl_func_name,
                config=self.config
            )
            
            # 7. 运行验证
            loop = asyncio.get_running_loop()
            multi_verify_res, multi_verify_log = await loop.run_in_executor(
                None,
                multi_case_verifier.run,
                self.conductor.task_info,  # task_info
                verify_step,                # current_step
                device_id                   # device_id
            )
            
            # 8. 处理验证结果
            if multi_verify_res:
                logger.info(f"[{self.op_name}] Hint模式多case验证通过")
                # 清除之前的错误信息
                self.conductor.task_info["multi_case_error"] = ""
                return True, ""
            else:
                logger.warning(f"[{self.op_name}] Hint模式多case验证失败")
                # 保存错误信息（供未来可能的错误分析使用）
                self.conductor.task_info["multi_case_error"] = multi_verify_log
                return False, multi_verify_log
                
        except Exception as e:
            logger.error(f"[{self.op_name}] Hint模式多case验证异常: {e}")
            import traceback
            error_log = traceback.format_exc()
            return False, error_log
    
    async def _run_llm_inference_mode_verification(self, device_id: int, verify_step: int) -> tuple[bool, str]:
        """LLM推理模式的多case验证"""
        try:
            # 1. 创建 TestCaseGenerator
            test_gen = TestCaseGenerator(
                op_name=self.op_name,
                task_desc=self.task_desc,
                framework=self.framework,
                dsl=self.dsl,
                config=self.config
            )
            
            # 1.1 准备输入数据（包含之前的错误信息）
            task_info_with_error = self.conductor.task_info.copy()
            # 获取之前的多 case 验证错误（如果有）
            previous_error = self.conductor.task_info.get("multi_case_error", "")
            task_info_with_error["previous_error"] = previous_error
            
            # 2. 生成多 case task_desc
            new_task_desc, prompt, reasoning = await test_gen.run(task_info_with_error)
            
            logger.info(f"[{self.op_name}] 多 case task_desc 生成完成")
            
            # 2.1 记录 TestCaseGenerator 的执行结果（保存 prompt 和返回值）
            self.conductor.record_agent_execution(
                agent_name="test_case_generator",
                result=new_task_desc,
                prompt=prompt,
                reasoning=reasoning
            )
            
            # 3. 使用新 task_desc 创建临时 verifier 进行验证
            # 注意：KernelVerifier 已经支持动态 shape 测试（通过 get_inputs_dyn_list）
            multi_case_verifier = KernelVerifier(
                op_name=self.op_name,
                framework_code=new_task_desc,  # 使用新的 task_desc
                task_id=f"{self.verifier.task_id}_multicase",
                framework=self.framework,
                dsl=self.dsl,
                backend=self.backend,
                arch=self.arch,
                impl_func_name=self.verifier.impl_func_name,
                config=self.config
            )
            
            # 使用与单 case 验证相同的 step（同一轮验证）
            loop = asyncio.get_running_loop()
            multi_verify_res, multi_verify_log = await loop.run_in_executor(
                None,
                multi_case_verifier.run,
                self.conductor.task_info,
                verify_step,  # 使用单 case 验证的 step
                device_id
            )
            
            # 4. 返回验证结果
            if multi_verify_res:
                logger.info(f"[{self.op_name}] 多 case 验证通过")
                # 清除之前的错误信息
                self.conductor.task_info["multi_case_error"] = ""
                return True, ""
            else:
                logger.warning(f"[{self.op_name}] 多 case 验证失败")
                # 保存错误信息到 task_info，供下次迭代使用
                self.conductor.task_info["multi_case_error"] = multi_verify_log
                return False, multi_verify_log
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{self.op_name}] 多 case 验证过程异常: {error_msg}")
            import traceback
            error_detail = traceback.format_exc()
            # 保存异常信息到 task_info，供下次迭代使用
            self.conductor.task_info["multi_case_error"] = f"多 case 验证异常: {error_msg}\n{error_detail}"
            return False, f"多 case 验证异常: {error_msg}\n{error_detail}"