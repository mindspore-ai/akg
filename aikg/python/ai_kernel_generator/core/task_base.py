# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in an writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Tuple
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.utils import check_task_config, check_task_type

logger = logging.getLogger(__name__)


class TaskBase:
    """
    任务基类
    """

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
        初始化任务基类，用于记录任务的各种属性。

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

        check_task_config(framework, backend, arch, impl_type)
        check_task_type(task_type)

        self.op_name = op_name
        self.task_desc = task_desc
        self.task_id = task_id
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.log_dir = config.get("log_dir")
        self.model_name_dict = config.get("agent_model_config")
        self.profile_settings = config.get("profile_settings")
        self.impl_type = impl_type
        self.framework = framework
        self.task_type = task_type
        self.limit_steps = limit_steps
        self.device_pool = device_pool

    async def run(self) -> Tuple[str, bool]:
        """
        运行任务
        """
        raise NotImplementedError("run method must be implemented in subclasses")
