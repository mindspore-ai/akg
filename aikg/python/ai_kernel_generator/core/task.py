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

import logging
import json
import asyncio
from typing import Tuple
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.utils import ActionType, ParsedCode
from ai_kernel_generator.core.agent.conductor import Conductor
from ai_kernel_generator.core.agent.aul_designer import AULDesigner
from ai_kernel_generator.core.agent.coder import CoderFactory
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.core.task_base import TaskBase

logger = logging.getLogger(__name__)


class Task(TaskBase):
    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 task_id: str,
                 backend: str,
                 arch: str,
                 impl_type: str,
                 config: dict,
                 device_pool: DevicePool,
                 framework: str,
                 task_type="precision_only",
                 limit_steps=10) -> None:
        """
        初始化任务类，用于记录任务的各种属性。

        Args:
            op_name (str): 算子名称。
            task_desc (str): 算子描述。
            task_id (str): 任务ID。
            backend (str): 后端名称。
            arch (str): 架构名称。
            impl_type (str): 实现类型。
            config (dict): 配置agent_mode_config。
            framework (str): 框架名称。
            task_type (str, optional): 任务类型, 默认为"precision_only"。
            limit_steps (int, optional): 限制步数, 默认为10。
        """
        super().__init__(op_name, task_desc, task_id, backend, arch, impl_type, config, device_pool, framework,
                         task_type, limit_steps)
        self.designer = AULDesigner(self.op_name, self.task_desc, self.model_name_dict,
                                    self.impl_type, self.backend, self.arch)
        self.coder = CoderFactory().create_coder(self.op_name, self.task_desc, self.model_name_dict, self.impl_type,
                                                 self.framework)
        self.verifier = KernelVerifier(self.op_name, self.task_desc, self.log_dir,
                                       self.task_id, self.framework, self.impl_type, self.backend, self.arch)
        self.conductor = Conductor(self.op_name, self.task_id, self.log_dir, self.impl_type, self.model_name_dict)

    def init_conductor(self, init_action_type=ActionType.DO_DESIGNER, init_parsed_code=ParsedCode()):
        """
        初始化Conductor，根据初始动作类型和解析代码进行初始化。
        Args:
            init_action_type (ActionType, optional): 初始动作类型, 默认为ActionType.DO_DESIGNER。
            init_parsed_code (ParsedCode, optional): 初始解析代码, 默认为ParsedCode()。
        Returns:
            None
        """
        # 初始化基础文档
        self.conductor.trace.base_doc.update(self.designer.aul_base_doc)
        if self.impl_type == "triton":
            self.conductor.trace.base_doc.update(self.coder.triton_base_doc)
        elif self.impl_type == "swft":
            self.conductor.trace.base_doc.update(self.coder.swft_base_doc)

        # 初始化检查文档
        self.conductor.initialize_check_docs()

        # 插入初始记录
        if init_action_type in (ActionType.DO_CODER, ActionType.VERIFY):
            self.conductor.trace.insert_designer_or_coder_record(
                json.dumps({"code": init_parsed_code.aul_code, "description": ""}), "", "", ActionType.DO_DESIGNER
            )
        if init_action_type == ActionType.VERIFY:
            if self.impl_type == "triton":
                self.conductor.trace.insert_designer_or_coder_record(
                    json.dumps({"code": init_parsed_code.triton_code, "description": ""}), "", "", ActionType.DO_CODER
                )
            elif self.impl_type == "swft":
                self.conductor.trace.insert_designer_or_coder_record(
                    json.dumps({"code": init_parsed_code.swft_code, "description": ""}), "", "", ActionType.DO_CODER
                )

    async def run(self, init_action_type=ActionType.DO_DESIGNER, init_parsed_code=ParsedCode(),
                  init_suggestions="") -> Tuple[str, bool]:
        """
        异步运行任务，执行操作任务字符串
        Args:
            init_action_type (ActionType, optional): 初始动作类型, 默认为ActionType.DO_DESIGNER。
            init_parsed_code (ParsedCode, optional): 初始解析代码, 默认为ParsedCode()。
            init_suggestions (str, optional): 初始建议, 默认为""。
        Returns:
            Tuple[str, bool]: 包含算子名称和验证结果的元组。
        """
        try:
            self.init_conductor(init_action_type, init_parsed_code)
            action_type = init_action_type
            parsed_code = init_parsed_code
            suggestions = init_suggestions

            verify_res = False

            while action_type != ActionType.EXIT:
                logger.info(f"Task {self.task_id}, op_name: {self.op_name}, action_type: {action_type.value}")
                if action_type in [ActionType.DO_DESIGNER, ActionType.FIX_DESIGNER]:
                    designer_res, designer_prompt, designer_reasoning = await self.designer.run(action_type, parsed_code, suggestions)
                    self.conductor.trace.insert_designer_or_coder_record(
                        designer_res, designer_prompt, designer_reasoning, action_type)
                elif action_type in [ActionType.DO_CODER, ActionType.FIX_CODER]:
                    coder_res, coder_prompt, coder_reasoning = await self.coder.run(action_type, parsed_code, suggestions)
                    if self.impl_type == "swft":
                        self.conductor.trace.base_doc.update(self.coder.intermediate_base_doc)
                    self.conductor.trace.insert_designer_or_coder_record(
                        coder_res, coder_prompt, coder_reasoning, action_type)
                elif action_type == ActionType.VERIFY:
                    device_id = await self.device_pool.acquire_device()
                    try:
                        current_step = len(self.conductor.trace.trace_list)
                        loop = asyncio.get_running_loop()
                        verify_res, verify_log = await loop.run_in_executor(
                            None,
                            self.verifier.run,
                            parsed_code, current_step, device_id
                        )
                        profile_res = ""
                        if verify_res and self.task_type == "profile" and self.backend == "ascend":
                            speedup = self.verifier.run_profile(current_step, device_id, self.profile_settings)
                            profile_res = f"speedup: {speedup:.6f}x"
                        self.conductor.trace.insert_verifier_record(
                            str(verify_res), verify_log, profile_res, action_type)
                    finally:
                        await self.device_pool.release_device(device_id)
                else:
                    raise ValueError(f"Unsupported target: {action_type}")

                action_type, parsed_code, suggestions = await self.conductor.get_next_action()
                if self.conductor.step >= self.limit_steps:
                    action_type = ActionType.EXIT

            if verify_res:
                return self.op_name, True
            else:
                return self.op_name, False
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return self.op_name, False
