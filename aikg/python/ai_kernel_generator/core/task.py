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
                 device_pool: DevicePool,
                 framework: str,
                 task_type="precision_only",
                 workflow: Optional[str] = None,
                 inspirations: Optional[List[str]] = None,) -> None:
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
            device_pool: 设备池。
            framework (str): 框架名称。
            task_type (str, optional): 任务类型, 默认为"precision_only"。
            workflow (str, optional): workflow名称，可以是文件名或完整路径。
                - 文件名: "coder_only_workflow" -> "config/coder_only_workflow.yaml"
                - 完整路径: "config/xxx.yaml"
                - None: 使用默认配置
            inspirations (List[str], optional): 启发示例列表。
        """
        # 验证任务配置
        check_task_config(framework, backend, arch, dsl)
        check_task_type(task_type)

        # 基础属性
        self.op_name = op_name
        self.task_desc = task_desc
        self.task_id = task_id
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.dsl = dsl.lower()
        self.framework = framework.lower()
        self.task_type = task_type
        self.device_pool = device_pool
        self.inspirations = inspirations

        # 统一保存config，后续向下传递
        self.config = config

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
        base_doc = {}

        # 只在相应agent存在时添加其基础文档
        if 'designer' in self.agents:
            base_doc.update(self.agents['designer'].base_doc)

        if 'coder' in self.agents:
            coder = self.agents['coder']
            base_doc.update(coder.base_doc)

        # 初始化任务信息
        self.conductor.set_task_info(base_doc)

        # inspirations from evolution
        self.conductor.task_info.update({"inspirations": self.inspirations})

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
                        device_id = await self.device_pool.acquire_device()
                        try:
                            current_step = len(self.conductor.trace.trace_list)
                            loop = asyncio.get_running_loop()
                            # 传递task_info而不是parsed_code
                            verify_res, verify_log = await loop.run_in_executor(
                                None,
                                self.verifier.run,
                                self.conductor.task_info, current_step, device_id
                            )
                            profile_res = ()
                            if verify_res and self.task_type == "profile" and self.backend in ["ascend", "cuda"]:
                                profile_settings = self.config.get("profile_settings", {})
                                profile_res = await loop.run_in_executor(
                                    None,
                                    self.verifier.run_profile,
                                    current_step, device_id, profile_settings
                                )

                            self.conductor.record_agent_execution(
                                agent_name="verifier",
                                result=str(verify_res),
                                error_log=verify_log,
                                profile_res=profile_res
                            )
                        finally:
                            await self.device_pool.release_device(device_id)
                    else:
                        raise ValueError(f"Unsupported agent: {current_agent}")

                    # 获取下一个agent
                    next_agent = await self.conductor.get_next_agent()
                    current_agent = next_agent

                except Exception as agent_error:
                    logger.error(f"Agent {current_agent} execution failed: {agent_error}")
                    self.conductor.record_agent_execution(
                        agent_name=current_agent,
                        result=f"ERROR: Agent execution failed: {str(agent_error)}",
                        error_log=str(agent_error)
                    )
                    return self.op_name, False, self.conductor.task_info

            # 获取最终结果
            final_success = self.conductor.task_info.get('verifier_result', False)
            return self.op_name, final_success, self.conductor.task_info

        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return self.op_name, False, self.conductor.task_info
