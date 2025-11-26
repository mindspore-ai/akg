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
                                profile_settings = self.config.get("profile_settings", {})
                                profile_res = await self.verifier.run_profile(
                                    self.conductor.task_info, current_step, profile_settings=profile_settings
                                )

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
