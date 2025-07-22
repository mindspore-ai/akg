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
import asyncio
from typing import Tuple
from ai_kernel_generator.core.task_base import TaskBase
from ai_kernel_generator.core.agent.coder import CoderFactory
from ai_kernel_generator.core.agent.conductor import Conductor
from ai_kernel_generator.core.verifier.kernel_verifier import KernelVerifier
from ai_kernel_generator.core.utils import ParsedCode, ActionType
from ai_kernel_generator.utils.common_utils import ParserFactory

logger = logging.getLogger(__name__)


class CoderOnlyTask(TaskBase):
    """
    仅包含Coder的任务类，用于单agent场景，只支持triton
    """

    def __init__(self,
                 op_name: str,
                 task_desc: str,
                 task_id: str,
                 backend: str,
                 arch: str,
                 impl_type: str,
                 config: dict,
                 device_pool,
                 framework: str,
                 task_type="precision_only",
                 limit_steps=6) -> None:
        """
        初始化CoderOnlyTask类。

        Args:
            op_name (str): 算子名称。
            task_desc (str): 算子描述。
            task_id (str): 任务ID。
            backend (str): 后端名称。
            arch (str): 架构名称。
            impl_type (str): 实现类型。
            config (dict): 配置agent_mode_config。
            device_pool: 设备池。
            framework (str): 框架名称。
            task_type (str, optional): 任务类型, 默认为"precision_only"。
            limit_steps (int, optional): 限制步数, 默认为10。
        """
        super().__init__(op_name, task_desc, task_id, backend, arch, impl_type, config, device_pool, framework,
                         task_type, limit_steps)

        # 只支持triton
        if "triton" not in impl_type:
            raise ValueError(f"CoderOnlyTask只支持triton，当前impl_type: {impl_type}")

        # 初始化coder
        self.coder = CoderFactory().create_coder(self.op_name, self.task_desc, self.model_name_dict,
                                                 self.impl_type, self.framework)

        # 初始化verifier
        self.verifier = KernelVerifier(self.op_name, self.task_desc, self.log_dir,
                                       self.task_id, self.framework, self.impl_type, self.backend, self.arch)

        # 初始化conductor (coder_only模式)
        self.conductor = Conductor(self.op_name, self.task_id, self.log_dir, self.impl_type,
                                   self.model_name_dict, coder_only_mode=True)

    def init_conductor(self):
        """
        初始化Conductor，针对coder_only模式进行优化。
        """
        # 初始化基础文档
        self.conductor.trace.base_doc.update(self.coder.triton_base_doc)

        # 初始化检查文档
        self.conductor.initialize_check_docs()

    async def run(self, init_action_type=ActionType.DO_CODER_DIRECT, init_parsed_code=ParsedCode(),
                  init_suggestions="") -> Tuple[str, bool]:
        """
        运行CoderOnlyTask，使用DO_CODER_DIRECT直接生成代码并验证。

        Args:
            init_action_type: 初始动作类型，默认为ActionType.DO_CODER_DIRECT
            init_parsed_code (ParsedCode, optional): 初始解析代码, 默认为ParsedCode()。
            init_suggestions (str, optional): 初始建议, 默认为""。

        Returns:
            Tuple[str, bool]: 包含算子名称和验证结果的元组。
        """
        try:
            logger.info(f"CoderOnlyTask {self.task_id}, op_name: {self.op_name}")

            # 初始化conductor
            self.init_conductor()

            current_action = init_action_type
            parsed_code = init_parsed_code
            suggestions = init_suggestions

            verify_res = False

            while current_action != ActionType.EXIT:
                logger.info(
                    f"CoderOnlyTask {self.task_id}, op_name: {self.op_name}, action_type: {current_action.value}")

                if current_action in [ActionType.DO_CODER_DIRECT, ActionType.FIX_CODER]:
                    # 运行coder
                    logger.info("Running coder...")

                    coder_res, coder_prompt, coder_reasoning = await self.coder.run(
                        action_type=current_action,
                        parsed_code=parsed_code,
                        informations=suggestions
                    )

                    # 插入coder记录到trace
                    self.conductor.trace.insert_designer_or_coder_record(
                        coder_res, coder_prompt, coder_reasoning, current_action
                    )

                    # 解析coder结果
                    try:
                        code_parser = ParserFactory.get_code_parser()
                        parsed_result = ParserFactory.robust_parse(coder_res, code_parser)

                        if not parsed_result or not hasattr(parsed_result, 'code'):
                            logger.error(f"Failed to parse coder result: {coder_res}")
                            return self.op_name, False

                        parsed_code.triton_code = parsed_result.code

                    except Exception as e:
                        logger.error(f"Failed to parse coder result: {e}, content: {coder_res}")
                        return self.op_name, False

                elif current_action == ActionType.VERIFY:
                    # 运行verifier
                    logger.info("Running verifier...")
                    device_id = await self.device_pool.acquire_device()
                    try:
                        current_step = len(self.conductor.trace.trace_list)
                        loop = asyncio.get_running_loop()
                        verify_res, verify_log = await loop.run_in_executor(
                            None,
                            self.verifier.run,
                            parsed_code, current_step, device_id
                        )

                        # 插入verifier记录到trace
                        profile_res = ""
                        if verify_res and self.task_type == "profile" and self.backend == "ascend":
                            speedup = self.verifier.run_profile(current_step, device_id, self.profile_settings)
                            profile_res = f"speedup: {speedup:.6f}x"
                            logger.info(f"Profile result: {profile_res}")

                        self.conductor.trace.insert_verifier_record(
                            str(verify_res), verify_log, profile_res, current_action
                        )

                    finally:
                        await self.device_pool.release_device(device_id)

                else:
                    logger.error(f"Unsupported action type: {current_action}")
                    break

                current_action, parsed_code, suggestions = await self.conductor.get_next_action()
                if self.conductor.step >= self.limit_steps:
                    current_action = ActionType.EXIT

            if verify_res:
                return self.op_name, True
            else:
                return self.op_name, False

        except Exception as e:
            logger.error(f"CoderOnlyTask {self.task_id} failed: {e}")
            return self.op_name, False
